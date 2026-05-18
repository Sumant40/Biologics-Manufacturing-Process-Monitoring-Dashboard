"""
app.py
------
Entry point for the Biologics Manufacturing MSPC Dashboard.

Run with:
    python src/dashboard/app.py

Then open http://localhost:8050 in your browser.
"""

import dash
import dash_bootstrap_components as dbc
from layout import build_layout
import os

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Bioprocess MSPC Dashboard",
)

app.layout = build_layout()

# Register callbacks
import callbacks  # noqa: F401, E402 — side-effect import

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
