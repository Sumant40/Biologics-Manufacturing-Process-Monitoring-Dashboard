"""
callbacks.py
------------
All Dash callback logic for the MSPC Dashboard.

Callbacks
---------
1. init_data          — On page load: run PCA/T²/SPE, store results
2. update_kpis        — Populate KPI cards from batch summary
3. update_t2_chart    — Render Hotelling's T² control chart
4. update_spe_chart   — Render SPE control chart
5. update_pca_scatter — PC1 vs PC2 score scatter by batch type
6. populate_dropdowns — Fill batch selector dropdowns
7. update_trend_chart — Time-series trend for selected batches + param
8. predict_yield      — XGBoost prediction from early-run features
9. update_importances — Feature importance bar chart
"""

import json
import numpy as np
import pandas as pd
import joblib
import yaml
import plotly.graph_objects as go
import plotly.express as px
import sqlalchemy as sa
from dash import Input, Output, State, callback, dash_table, html
import dash_bootstrap_components as dbc
from pathlib import Path
import sys

# Path setup
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))

from monitoring.pca_engine import PCAEngine
from monitoring.hotelling_t2 import HotellingT2
from monitoring.spe import SPEMonitor
from modeling.feature_engineering import extract_early_features

CONFIG_PATH = ROOT / "config" / "params.yaml"
MODEL_DIR = ROOT / "data" / "processed"

config = yaml.safe_load(open(CONFIG_PATH))
DB_PATH = ROOT / config["database"]["path"]
engine_db = sa.create_engine(f"sqlite:///{DB_PATH}")

PARAMS = [p for p in config["process_parameters"] if p != "titer_mg_L"]
PALETTE = {"normal": "#2ecc71", "marginal": "#f39c12", "ooc": "#e74c3c"}


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_meas = pd.read_sql("SELECT * FROM measurements", engine_db)
    df_batches = pd.read_sql("SELECT * FROM batches", engine_db)
    return df_meas, df_batches


# ── Callback 1: Initialise MSPC on page load ───────────────────────

@callback(
    Output("store-t2-data", "data"),
    Output("store-spe-data", "data"),
    Output("store-pca-scores", "data"),
    Output("store-batch-list", "data"),
    Input("init-trigger", "n_intervals"),
    prevent_initial_call=False,
)
def init_data(_):
    df_meas, df_batches = get_data()
    df_all = df_meas.merge(df_batches[["batch_id", "batch_type"]], on="batch_id")
    df_clean = df_all.dropna(subset=PARAMS)

    X_all = df_clean[PARAMS].values
    batch_ids = df_clean["batch_id"].tolist()
    batch_types = df_clean["batch_type"].tolist()

    # Load or fit PCA
    try:
        pca_engine = PCAEngine.load()
    except Exception:
        pca_engine = PCAEngine(n_components=config["mspc"]["n_components"])
        mask_ref = df_clean["batch_type"] == "normal"
        pca_engine.fit(X_all[mask_ref])
        pca_engine.save()

    T_all, spe_all = pca_engine.transform(X_all)

    # Load or fit T² monitor
    try:
        t2_monitor = HotellingT2.load()
    except Exception:
        mask_ref = [bt == "normal" for bt in batch_types]
        T_ref = T_all[np.array(mask_ref)]
        t2_monitor = HotellingT2(
            n_components=config["mspc"]["n_components"],
            alpha=config["mspc"]["alpha"],
        )
        t2_monitor.fit(T_ref)
        t2_monitor.save()

    t2_values = t2_monitor.score(T_all)
    t2_summary = t2_monitor.summary(T_all, batch_ids)

    # Batch-level T² (max per batch)
    t2_data = t2_summary.merge(
        df_batches[["batch_id", "batch_type"]], on="batch_id"
    ).to_dict("records")

    # SPE per batch (max)
    spe_df = pd.DataFrame({"batch_id": batch_ids, "spe": spe_all})
    spe_agg = (
        spe_df.groupby("batch_id")["spe"].max().reset_index()
        .rename(columns={"spe": "spe_max"})
        .merge(df_batches[["batch_id", "batch_type"]], on="batch_id")
    )
    spe_data = spe_agg.to_dict("records")

    # PCA scores (first two PCs per time-point, sampled for performance)
    sample_mask = np.random.choice(len(T_all), min(2000, len(T_all)), replace=False)
    pca_scores = {
        "pc1": T_all[sample_mask, 0].tolist(),
        "pc2": T_all[sample_mask, 1].tolist(),
        "batch_type": [batch_types[i] for i in sample_mask],
        "batch_id": [batch_ids[i] for i in sample_mask],
        "ucl_t2": t2_monitor.ucl_phase2_,
    }

    batch_list = df_batches[["batch_id", "batch_type"]].to_dict("records")

    return t2_data, spe_data, pca_scores, batch_list


# ── Callback 2: KPI cards ──────────────────────────────────────────

@callback(
    Output("kpi-total", "children"),
    Output("kpi-normal", "children"),
    Output("kpi-marginal", "children"),
    Output("kpi-ooc", "children"),
    Input("store-batch-list", "data"),
)
def update_kpis(batch_list):
    if not batch_list:
        return "—", "—", "—", "—"
    df = pd.DataFrame(batch_list)
    counts = df["batch_type"].value_counts()
    return (
        str(len(df)),
        str(counts.get("normal", 0)),
        str(counts.get("marginal", 0)),
        str(counts.get("ooc", 0)),
    )


# ── Callback 3: Batch overview table ──────────────────────────────

@callback(
    Output("batch-table", "children"),
    Input("store-t2-data", "data"),
)
def update_batch_table(t2_data):
    if not t2_data:
        return "Loading…"
    df = pd.DataFrame(t2_data)
    df["Status"] = df.apply(
        lambda r: "🔴 OOC" if r["ooc"] else ("🟡 Marginal" if r["batch_type"] == "marginal" else "🟢 Normal"),
        axis=1,
    )
    display_df = df[["batch_id", "batch_type", "t2_max", "t2_mean", "ucl", "Status"]].copy()
    display_df.columns = ["Batch ID", "Type", "T² Max", "T² Mean", "UCL", "Status"]
    display_df = display_df.round(2)

    return dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display_df.columns],
        sort_action="native",
        filter_action="native",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#2c3e50", "color": "white"},
        style_data_conditional=[
            {"if": {"filter_query": '{Status} = "🔴 OOC"'}, "backgroundColor": "#fadbd8"},
            {"if": {"filter_query": '{Status} = "🟡 Marginal"'}, "backgroundColor": "#fef9e7"},
        ],
    )


# ── Callback 4: T² Chart ───────────────────────────────────────────

@callback(
    Output("t2-chart", "figure"),
    Input("store-t2-data", "data"),
)
def update_t2_chart(t2_data):
    if not t2_data:
        return go.Figure()
    df = pd.DataFrame(t2_data).sort_values("t2_max", ascending=False).reset_index(drop=True)

    fig = go.Figure()
    for btype, color in PALETTE.items():
        mask = df["batch_type"] == btype
        fig.add_trace(go.Bar(
            x=df[mask].index,
            y=df[mask]["t2_max"],
            name=btype.capitalize(),
            marker_color=color,
            text=df[mask]["batch_id"],
            hovertemplate="<b>%{text}</b><br>T² Max: %{y:.2f}<extra></extra>",
        ))

    ucl = df["ucl"].iloc[0]
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL = {ucl:.1f}", annotation_position="top right")

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Batch (ranked by T² max)",
        yaxis_title="Hotelling's T² (max)",
        template="plotly_dark",
        legend_title="Batch Type",
        height=380,
    )
    return fig


# ── Callback 5: SPE Chart ──────────────────────────────────────────

@callback(
    Output("spe-chart", "figure"),
    Input("store-spe-data", "data"),
)
def update_spe_chart(spe_data):
    if not spe_data:
        return go.Figure()
    df = pd.DataFrame(spe_data).sort_values("spe_max", ascending=False).reset_index(drop=True)

    fig = go.Figure()
    for btype, color in PALETTE.items():
        mask = df["batch_type"] == btype
        fig.add_trace(go.Bar(
            x=df[mask].index,
            y=df[mask]["spe_max"],
            name=btype.capitalize(),
            marker_color=color,
            text=df[mask]["batch_id"],
            hovertemplate="<b>%{text}</b><br>SPE: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Batch",
        yaxis_title="SPE Max",
        template="plotly_dark",
        height=360,
        showlegend=False,
    )
    return fig


# ── Callback 6: PCA Score Scatter ─────────────────────────────────

@callback(
    Output("pca-scatter", "figure"),
    Input("store-pca-scores", "data"),
)
def update_pca_scatter(pca_data):
    if not pca_data:
        return go.Figure()

    fig = go.Figure()
    types = set(pca_data["batch_type"])
    pc1 = np.array(pca_data["pc1"])
    pc2 = np.array(pca_data["pc2"])
    btypes = np.array(pca_data["batch_type"])

    for btype in ["normal", "marginal", "ooc"]:
        if btype not in types:
            continue
        mask = btypes == btype
        fig.add_trace(go.Scatter(
            x=pc1[mask], y=pc2[mask],
            mode="markers",
            marker=dict(color=PALETTE[btype], size=5, opacity=0.6),
            name=btype.capitalize(),
        ))

    fig.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_dark",
        height=360,
        legend_title="Batch Type",
    )
    return fig


# ── Callback 7: Populate batch dropdowns ──────────────────────────

@callback(
    Output("trend-batch-select", "options"),
    Output("pred-batch-select", "options"),
    Input("store-batch-list", "data"),
)
def populate_dropdowns(batch_list):
    if not batch_list:
        return [], []
    opts = [{"label": f"{b['batch_id']} ({b['batch_type']})", "value": b["batch_id"]}
            for b in batch_list]
    return opts, opts


# ── Callback 8: Process Trend Chart ───────────────────────────────

@callback(
    Output("trend-chart", "figure"),
    Input("trend-batch-select", "value"),
    Input("trend-param-select", "value"),
)
def update_trend_chart(batch_ids, param):
    if not batch_ids or not param:
        return go.Figure()

    ids_str = ", ".join([f"'{b}'" for b in batch_ids])
    df = pd.read_sql(
        f"SELECT batch_id, time_h, {param} FROM measurements WHERE batch_id IN ({ids_str})",
        engine_db,
    )

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, bid in enumerate(batch_ids):
        sub = df[df["batch_id"] == bid].sort_values("time_h")
        fig.add_trace(go.Scatter(
            x=sub["time_h"],
            y=sub[param],
            mode="lines+markers",
            name=bid,
            marker=dict(size=4),
            line=dict(color=colors[i % len(colors)]),
        ))

    fig.update_layout(
        xaxis_title="Time (h)",
        yaxis_title=param,
        template="plotly_dark",
        height=420,
        legend_title="Batch",
    )
    return fig


# ── Callback 9: Yield Prediction ───────────────────────────────────

@callback(
    Output("pred-output", "children"),
    Input("pred-btn", "n_clicks"),
    State("pred-batch-select", "value"),
    prevent_initial_call=True,
)
def predict_yield(n_clicks, batch_id):
    if not batch_id:
        return dbc.Alert("Please select a batch.", color="warning")

    try:
        model = joblib.load(MODEL_DIR / "yield_model.joblib")
    except FileNotFoundError:
        return dbc.Alert("Yield model not found. Run train_yield_model.py first.", color="danger")

    df_meas = pd.read_sql(
        f"SELECT * FROM measurements WHERE batch_id = '{batch_id}'",
        engine_db,
    )
    cutoff = config["modeling"]["early_run_cutoff_hours"]
    df_feat = extract_early_features(df_meas, cutoff_h=cutoff, params=PARAMS)

    feature_cols = [c for c in df_feat.columns if "__" in c]
    X = df_feat[feature_cols].fillna(0).values

    y_pred = model.predict(X)[0]

    # Actual titer for comparison
    actual = pd.read_sql(
        f"SELECT final_titer_mg_L FROM batches WHERE batch_id = '{batch_id}'",
        engine_db,
    ).iloc[0, 0]

    return dbc.Card([
        dbc.CardBody([
            html.H4(f"Batch: {batch_id}", className="card-title"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Predicted Titer (from 48h data)", className="text-muted mb-1"),
                    html.H2(f"{y_pred:.1f} mg/L", className="text-primary"),
                ], md=6),
                dbc.Col([
                    html.P("Actual Final Titer", className="text-muted mb-1"),
                    html.H2(f"{actual:.1f} mg/L", className="text-success"),
                ], md=6),
            ]),
            html.P(f"Error: {abs(y_pred - actual):.1f} mg/L ({abs(y_pred-actual)/actual*100:.1f}%)",
                   className="text-muted mt-2"),
        ])
    ], className="border-primary")


# ── Callback 10: Feature Importance Chart ─────────────────────────

@callback(
    Output("importance-chart", "figure"),
    Input("init-trigger", "n_intervals"),
)
def update_importances(_):
    imp_path = MODEL_DIR / "feature_importances.csv"
    if not imp_path.exists():
        return go.Figure()

    df = pd.read_csv(imp_path).head(20).sort_values("importance_gain")
    fig = go.Figure(go.Bar(
        x=df["importance_gain"],
        y=df["feature"],
        orientation="h",
        marker_color="#3498db",
    ))
    fig.update_layout(
        xaxis_title="Importance (Gain)",
        yaxis_title="",
        template="plotly_dark",
        height=480,
        margin=dict(l=200),
    )
    return fig
