"""
Kronos adapter.

Optional, lazily-loaded interface to the Kronos foundation model for
financial candlesticks (https://github.com/shiyu-coder/Kronos). This module
is **safe to import** at app startup even when Kronos is not installed —
every heavy dependency (`torch`, the Kronos source code, Hugging Face
client) is imported inside the function that needs it.

The integration follows the official Kronos usage pattern verbatim:
    tokenizer = KronosTokenizer.from_pretrained(<tokenizer_id>)
    model     = Kronos.from_pretrained(<model_id>)
    predictor = KronosPredictor(model, tokenizer, device=..., max_context=...)
    pred_df   = predictor.predict(df=x_df, x_timestamp=..., y_timestamp=...,
                                  pred_len=..., T=..., top_p=..., sample_count=...)

Source code resolution order (ascending priority):
  1. KRONOS_REPO_PATH environment variable
  2. <repo>/external/Kronos
  3. <repo parent>/Kronos

Model weights are pulled from Hugging Face on first use and cached by the
local `huggingface_hub` cache. We do NOT download or commit weights.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .. import config, utils

logger = utils.get_logger("cte.kronos")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KronosModelSpec:
    name: str
    model_id: str
    tokenizer_id: str
    max_context: int


KRONOS_MODELS: Dict[str, KronosModelSpec] = {
    "Kronos-mini": KronosModelSpec(
        name="Kronos-mini",
        model_id="NeoQuasar/Kronos-mini",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",
        max_context=2048,
    ),
    "Kronos-small": KronosModelSpec(
        name="Kronos-small",
        model_id="NeoQuasar/Kronos-small",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
    ),
    "Kronos-base": KronosModelSpec(
        name="Kronos-base",
        model_id="NeoQuasar/Kronos-base",
        tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
    ),
}

DEFAULT_MODEL = "Kronos-mini"
DEFAULT_LOOKBACK = 400
DEFAULT_PRED_LEN = 24
DEFAULT_T = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_SAMPLE_COUNT = 1


# ---------------------------------------------------------------------------
# Repo + dependency resolution (lazy)
# ---------------------------------------------------------------------------
def _candidate_repo_paths() -> List[Path]:
    """Return the ordered list of paths Kronos source code might live in."""
    repo_root = config.REPO_ROOT
    candidates: List[Path] = []
    env = os.environ.get("KRONOS_REPO_PATH")
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append((repo_root / "external" / "Kronos").resolve())
    candidates.append((repo_root.parent / "Kronos").resolve())
    return candidates


def _resolve_kronos_repo_path() -> Optional[Path]:
    for cand in _candidate_repo_paths():
        if cand.exists() and (cand / "model").exists():
            return cand
    return None


def _ensure_repo_on_path() -> Optional[Path]:
    """Append the resolved Kronos repo to sys.path if found. Idempotent."""
    repo = _resolve_kronos_repo_path()
    if repo is None:
        return None
    s = str(repo)
    if s not in sys.path:
        sys.path.insert(0, s)
    return repo


def _check_python_deps() -> Tuple[bool, List[str]]:
    """Return (all_present, missing_list) without importing the heavy stuff
    at module level."""
    required = ("torch", "transformers", "huggingface_hub",
                "einops", "safetensors")
    missing: List[str] = []
    for mod in required:
        try:
            import_module(mod)
        except Exception:  # noqa: BLE001
            missing.append(mod)
    return (len(missing) == 0), missing


def kronos_available() -> bool:
    """Quick yes/no for the dashboard. Never raises."""
    deps_ok, _ = _check_python_deps()
    if not deps_ok:
        return False
    repo = _resolve_kronos_repo_path()
    if repo is None:
        return False
    # Try to import the Kronos symbols themselves.
    try:
        _ensure_repo_on_path()
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def get_kronos_status() -> Dict[str, object]:
    """Detailed status for diagnostics. Never raises."""
    deps_ok, missing = _check_python_deps()
    repo = _resolve_kronos_repo_path()
    candidates = [str(p) for p in _candidate_repo_paths()]
    available = False
    import_error: Optional[str] = None
    if deps_ok and repo is not None:
        try:
            _ensure_repo_on_path()
            from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore  # noqa: F401
            available = True
        except Exception as e:  # noqa: BLE001
            import_error = f"{type(e).__name__}: {e}"
    return {
        "available": available,
        "python_deps_ok": deps_ok,
        "missing_python_deps": missing,
        "repo_path_resolved": str(repo) if repo else None,
        "repo_path_candidates_checked": candidates,
        "kronos_repo_path_env": os.environ.get("KRONOS_REPO_PATH"),
        "import_error": import_error,
        "supported_models": list(KRONOS_MODELS.keys()),
        "default_model": DEFAULT_MODEL,
    }


# ---------------------------------------------------------------------------
# Forecast input preparation
# ---------------------------------------------------------------------------
_TIMEFRAME_MS: Dict[str, int] = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def _timeframe_to_ms(timeframe: str) -> int:
    if timeframe not in _TIMEFRAME_MS:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}. Supported: "
            f"{sorted(_TIMEFRAME_MS)}"
        )
    return _TIMEFRAME_MS[timeframe]


def prepare_forecast_inputs(
    candles: pd.DataFrame,
    timeframe: str,
    lookback: int = DEFAULT_LOOKBACK,
    pred_len: int = DEFAULT_PRED_LEN,
    end_index: Optional[int] = None,
    max_context: int = KRONOS_MODELS[DEFAULT_MODEL].max_context,
) -> Dict[str, object]:
    """Slice the most recent `lookback` candles ending at `end_index`
    (default: last row), and emit Kronos-ready inputs.

    Returns a dict with `x_df`, `x_timestamp`, `y_timestamp`, `pred_len`.

    The caller — and Kronos itself — should reject calls where the slice is
    too short. We never silently pad. We also clamp `lookback` to the
    model's `max_context`.
    """
    if "datetime" in candles.columns:
        df = candles.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in candles.columns:
        df = candles.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError("candles must have a 'datetime' or 'timestamp' column")

    required = ("open", "high", "low", "close")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"candles missing required column(s): {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = 0.0

    if end_index is None:
        end_index = len(df)
    if end_index > len(df):
        end_index = len(df)
    lookback = min(int(lookback), int(max_context))
    start = max(0, end_index - lookback)
    if (end_index - start) < lookback:
        raise ValueError(
            f"not enough candles for lookback {lookback}: have {end_index - start}"
        )

    x_df = df.iloc[start:end_index][
        ["open", "high", "low", "close", "volume", "amount"]
    ].copy().reset_index(drop=True)
    x_timestamp = pd.Series(
        df.iloc[start:end_index]["datetime"].values, name="x_timestamp",
    ).reset_index(drop=True)

    tf_ms = _timeframe_to_ms(timeframe)
    last_ts = pd.Timestamp(df.iloc[end_index - 1]["datetime"])
    future = pd.date_range(
        start=last_ts + pd.Timedelta(milliseconds=tf_ms),
        periods=int(pred_len),
        freq=pd.Timedelta(milliseconds=tf_ms),
        tz="UTC",
    )
    y_timestamp = pd.Series(future, name="y_timestamp")
    return {
        "x_df": x_df,
        "x_timestamp": x_timestamp,
        "y_timestamp": y_timestamp,
        "pred_len": int(pred_len),
    }


# ---------------------------------------------------------------------------
# Predictor loading + forecast
# ---------------------------------------------------------------------------
def load_kronos_predictor(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
):
    """Construct a `KronosPredictor` using the official Kronos source.

    Lazy import of `torch` and Kronos — never called at app startup.
    Raises a clean `RuntimeError` if dependencies are missing so callers can
    surface the install instructions.
    """
    if model_name not in KRONOS_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Supported: {list(KRONOS_MODELS)}."
        )
    if model_name == "Kronos-large":  # extra safety; never in registry
        raise ValueError("Kronos-large is not supported.")

    deps_ok, missing = _check_python_deps()
    if not deps_ok:
        raise RuntimeError(
            f"Optional ML dependencies missing: {missing}. "
            "Install with: pip install -r requirements-ml.txt"
        )
    if _ensure_repo_on_path() is None:
        raise RuntimeError(
            "Kronos source code not found. Clone "
            "https://github.com/shiyu-coder/Kronos into external/Kronos "
            "or set KRONOS_REPO_PATH."
        )
    spec = KRONOS_MODELS[model_name]
    # Lazy imports — anything that pulls torch happens only here.
    from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
    logger.info("loading kronos %s on %s", spec.name, device)
    tokenizer = KronosTokenizer.from_pretrained(spec.tokenizer_id)
    model = Kronos.from_pretrained(spec.model_id)
    predictor = KronosPredictor(
        model, tokenizer, device=device, max_context=spec.max_context,
    )
    return predictor, spec


def run_kronos_forecast(
    candles: pd.DataFrame,
    timeframe: str,
    model_name: str = DEFAULT_MODEL,
    lookback: int = DEFAULT_LOOKBACK,
    pred_len: int = DEFAULT_PRED_LEN,
    device: str = "cpu",
    T: float = DEFAULT_T,
    top_p: float = DEFAULT_TOP_P,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    end_index: Optional[int] = None,
    predictor: Optional[Any] = None,
) -> pd.DataFrame:
    """End-to-end single forecast. Returns the forecast dataframe with the
    same `open/high/low/close[/volume/amount]` columns Kronos emits.

    Pass `predictor` to reuse a loaded model across windows; otherwise it
    is loaded fresh (slow on first call due to model download).
    """
    spec = KRONOS_MODELS[model_name]
    inputs = prepare_forecast_inputs(
        candles, timeframe=timeframe, lookback=lookback,
        pred_len=pred_len, end_index=end_index, max_context=spec.max_context,
    )
    if predictor is None:
        predictor, _ = load_kronos_predictor(model_name=model_name, device=device)
    pred_df = predictor.predict(
        df=inputs["x_df"],
        x_timestamp=inputs["x_timestamp"],
        y_timestamp=inputs["y_timestamp"],
        pred_len=inputs["pred_len"],
        T=T, top_p=top_p, sample_count=sample_count,
    )
    return pred_df
