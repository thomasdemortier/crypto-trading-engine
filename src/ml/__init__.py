"""
Optional ML research layer.

This package is purely opt-in. Importing it (or any of its submodules) does
NOT pull in `torch`, `transformers`, or any Hugging Face client at module
top level — every heavy import lives inside the function that needs it. The
core app, the dashboard, the backtester, the risk engine, and Streamlit
Cloud deploys all keep working with `requirements.txt` alone.

Kronos integration philosophy:
  * Kronos NEVER trades.
  * Kronos NEVER bypasses the risk engine.
  * Kronos is a CONFIRMATION layer for rule-based signals only:
      strategy proposes -> Kronos confirms/rejects -> risk engine decides.

To enable Kronos locally:
  pip install -r requirements-ml.txt
  git clone https://github.com/shiyu-coder/Kronos.git external/Kronos
  python main.py kronos_status
"""
