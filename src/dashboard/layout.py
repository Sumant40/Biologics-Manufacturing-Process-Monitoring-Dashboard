"""
layout.py
---------
Defines the full Dash page structure.

Tabs
----
1. Batch Overview     — KPI cards + batch status table (normal/marginal/OOC)
2. MSPC Control Charts — T² control chart, SPE chart, PCA score scatter
3. Process Trends     — Time-series for any parameter, any batch
4. Yield Predictor    — Early-run feature inputs → XGBoost yield estimate

All figures are populated by callbacks.py via dcc.Store data sharing.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


def kpi_card(title: str, value_id: str, color: str = "primary") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted mb-1"),
            html.H3(id=value_id, className="card-title mb-0"),
        ]),
        color=color,
        inverse=True,
        className="shadow-sm",
    )


def build_layout() -> html.Div:
    return html.Div([

        # ── Top navigation bar ─────────────────────────────────────────
        dbc.Navbar(
            dbc.Container([
                html.Span("⚗️", style={"fontSize": "1.5rem", "marginRight": "8px"}),
                dbc.NavbarBrand("Bioprocess MSPC Dashboard", className="fw-bold"),
                html.Small(" | Drug Substance Process Development", className="text-muted"),
            ], fluid=True),
            color="dark",
            dark=True,
            className="mb-3",
        ),

        dbc.Container([

            # ── KPI Row ────────────────────────────────────────────────
            dbc.Row([
                dbc.Col(kpi_card("Total Batches", "kpi-total", "secondary"), md=3),
                dbc.Col(kpi_card("Normal", "kpi-normal", "success"), md=3),
                dbc.Col(kpi_card("Marginal", "kpi-marginal", "warning"), md=3),
                dbc.Col(kpi_card("Out-of-Control", "kpi-ooc", "danger"), md=3),
            ], className="mb-4"),

            # ── Tab structure ──────────────────────────────────────────
            dbc.Tabs([

                # Tab 1: Batch Overview
                dbc.Tab(label="📋 Batch Overview", tab_id="tab-overview", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5("Batch Status Summary", className="mt-3 mb-2"),
                            dcc.Loading(
                                html.Div(id="batch-table"),
                                type="circle",
                            ),
                        ])
                    ])
                ]),

                # Tab 2: MSPC Control Charts
                dbc.Tab(label="📈 MSPC Charts", tab_id="tab-mspc", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5("Hotelling's T² Control Chart", className="mt-3 mb-1"),
                            html.Small(
                                "Points above the UCL indicate out-of-control process behaviour.",
                                className="text-muted",
                            ),
                            dcc.Loading(dcc.Graph(id="t2-chart"), type="circle"),
                        ], md=12),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.H5("SPE (Q-statistic) Chart", className="mt-2 mb-1"),
                            dcc.Loading(dcc.Graph(id="spe-chart"), type="circle"),
                        ], md=6),
                        dbc.Col([
                            html.H5("PCA Score Plot (PC1 vs PC2)", className="mt-2 mb-1"),
                            dcc.Loading(dcc.Graph(id="pca-scatter"), type="circle"),
                        ], md=6),
                    ]),
                ]),

                # Tab 3: Process Trends
                dbc.Tab(label="📉 Process Trends", tab_id="tab-trends", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Batch:", className="fw-bold mt-3"),
                            dcc.Dropdown(id="trend-batch-select", multi=True, placeholder="Choose batches…"),
                        ], md=6),
                        dbc.Col([
                            html.Label("Select Parameter:", className="fw-bold mt-3"),
                            dcc.Dropdown(
                                id="trend-param-select",
                                options=[
                                    {"label": p, "value": p}
                                    for p in [
                                        "pH", "DO_pct", "temperature_C", "agitation_rpm",
                                        "feed_rate_A_mL_h", "feed_rate_B_mL_h",
                                        "glucose_g_L", "lactate_g_L",
                                        "VCD_e6_mL", "viability_pct", "titer_mg_L",
                                    ]
                                ],
                                value="VCD_e6_mL",
                            ),
                        ], md=6),
                    ]),
                    dcc.Loading(dcc.Graph(id="trend-chart"), type="circle"),
                ]),

                # Tab 4: Yield Predictor
                dbc.Tab(label="🔮 Yield Predictor", tab_id="tab-predictor", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5("Early-Run Yield Prediction (XGBoost)", className="mt-3 mb-1"),
                            html.P(
                                "Select a batch to predict final titer from its first 48h of data.",
                                className="text-muted",
                            ),
                            html.Label("Select Batch:", className="fw-bold"),
                            dcc.Dropdown(id="pred-batch-select", placeholder="Choose a batch…"),
                            dbc.Button(
                                "Predict Yield",
                                id="pred-btn",
                                color="primary",
                                className="mt-3",
                            ),
                            html.Div(id="pred-output", className="mt-4"),
                        ], md=5),
                        dbc.Col([
                            html.H5("Feature Importances (Top 20)", className="mt-3 mb-1"),
                            dcc.Loading(dcc.Graph(id="importance-chart"), type="circle"),
                        ], md=7),
                    ]),
                ]),

            ], id="tabs", active_tab="tab-overview"),

        ], fluid=True),

        # Hidden data stores
        dcc.Store(id="store-t2-data"),
        dcc.Store(id="store-spe-data"),
        dcc.Store(id="store-pca-scores"),
        dcc.Store(id="store-batch-list"),

        # Trigger initial load
        dcc.Interval(id="init-trigger", interval=1, max_intervals=1),

    ])
