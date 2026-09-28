"""
Load-Shedding Risk Prediction — Dash front-end
Loads the real trained model (final_model.pkl) and serves it on port 8050.
"""
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import json
from datetime import datetime
from pathlib import Path

import joblib
import plotly.graph_objects as go
import numpy as np
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

# ---------------------------------------------------------------------------
# Artifacts (same folder as this file)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent

MODEL     = joblib.load(BASE / "final_model.pkl")
SCALER    = joblib.load(BASE / "scaler.pkl")
FEATURES  = json.load(open(BASE / "feature_names.json"))       # list of 97 names
THRESHOLD = json.load(open(BASE / "threshold.json"))["threshold"]

# A "neutral" baseline for every feature we don't ask the user for: the
# training mean. After StandardScaler, these become z = 0.0, i.e. the model
# treats those features as perfectly typical.
try:
    NEUTRAL = dict(zip(FEATURES, np.asarray(SCALER.mean_).tolist()))
except AttributeError:
    NEUTRAL = {f: 0.0 for f in FEATURES}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def risk_band(prob: float) -> str:
    if prob < THRESHOLD:
        return "Low"
    if prob < 0.5:
        return "Medium"
    return "High"


def predict(raw: dict):
    """raw = the 11 form inputs. Returns (probability, risk_label)."""
    p = dict(NEUTRAL)              # start from neutral
    p.update(raw)                  # override with the user's inputs

    # ---- Time features from "now" ----
    now = datetime.now()
    p["hour"]         = now.hour
    p["day_of_week"]  = now.weekday()
    p["month"]        = now.month
    p["quarter"]      = (now.month - 1) // 3 + 1
    p["year"]         = now.year
    p["day_of_year"]  = now.timetuple().tm_yday
    p["is_weekend"]   = int(now.weekday() >= 5)
    p["season_summer"] = int(now.month in (12, 1, 2))
    p["season_autumn"] = int(now.month in (3, 4, 5))
    p["season_winter"] = int(now.month in (6, 7, 8))
    p["season_spring"] = int(now.month in (9, 10, 11))

    # ---- Derived lags & rolling windows ----
    demand = float(p.get("residual_demand", 25000.0))
    temp   = float(p.get("temperature_2m", 22.5))
    wind   = float(p.get("wind_speed_10m", 5.2))
    rad    = float(p.get("shortwave_radiation", 500.0))

    p["temp_hour_interaction"] = temp * p["hour"]

    for lag, mult in [(1, 0.99), (2, 0.98), (24, 1.00)]:
        p[f"residual_demand_lag{lag}"] = demand * mult
    p["residual_demand_roll_mean_6h"]  = demand * 0.99
    p["residual_demand_roll_max_6h"]   = demand * 1.05
    p["residual_demand_roll_min_6h"]   = demand * 0.95
    p["residual_demand_roll_mean_24h"] = demand * 0.98
    p["residual_demand_roll_max_24h"]  = demand * 1.10
    p["residual_demand_roll_min_24h"]  = demand * 0.90

    for lag in (1, 2, 24):
        p[f"temperature_2m_lag{lag}"] = temp
    p["temperature_2m_roll_mean_6h"]  = temp
    p["temperature_2m_roll_max_6h"]   = temp + 1.5
    p["temperature_2m_roll_min_6h"]   = temp - 1.5
    p["temperature_2m_roll_mean_24h"] = temp
    p["temperature_2m_roll_max_24h"]  = temp + 3.0
    p["temperature_2m_roll_min_24h"]  = temp - 3.0

    for lag in (1, 2, 24):
        p[f"wind_speed_10m_lag{lag}"] = wind
    p["wind_speed_10m_roll_mean_6h"]  = wind
    p["wind_speed_10m_roll_max_6h"]   = wind * 1.3
    p["wind_speed_10m_roll_min_6h"]   = max(wind * 0.5, 0.0)
    p["wind_speed_10m_roll_mean_24h"] = wind
    p["wind_speed_10m_roll_max_24h"]  = wind * 1.5
    p["wind_speed_10m_roll_min_24h"]  = max(wind * 0.3, 0.0)

    for lag in (1, 2, 24):
        p[f"shortwave_radiation_lag{lag}"] = rad
    p["shortwave_radiation_roll_mean_6h"]  = rad * 0.95
    p["shortwave_radiation_roll_max_6h"]   = rad * 1.20
    p["shortwave_radiation_roll_min_6h"]   = rad * 0.50
    p["shortwave_radiation_roll_mean_24h"] = rad * 0.70
    p["shortwave_radiation_roll_max_24h"]  = rad * 1.20
    p["shortwave_radiation_roll_min_24h"]  = 0.0

    # ---- Assemble in the exact order the scaler was fit on ----
    vec = np.array([[p.get(f, 0.0) for f in FEATURES]], dtype=float)
    scaled = SCALER.transform(vec)
    prob = float(MODEL.predict_proba(scaled)[0, 1])
    return prob, risk_band(prob)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
INPUT_GROUPS = [
    ("Demand & Supply Balance", [
        ("input-residual-demand",     "Residual Demand",              "MW",   25000.0, 4),
        ("input-rsa-contracted",      "RSA Contracted Forecast",      "MW",   21000.0, 4),
        ("input-dispatchable-gen",    "Dispatchable Generation",      "MW",   21000.0, 4),
    ]),
    ("Generation Mix", [
        ("input-thermal-gen",         "Thermal Generation",           "MW",   21000.0, 4),
        ("input-nuclear-gen",         "Nuclear Generation",           "MW",   1800.0,  4),
        ("input-total-re",            "Total Renewable Energy",       "MW",   2300.0,  4),
    ]),
    ("Weather Conditions", [
        ("input-temperature",         "Temperature (2m)",             "°C",   22.5,    3),
        ("input-wind-speed",          "Wind Speed (10m)",             "m/s",  5.2,     3),
        ("input-cloud-cover",         "Cloud Cover",                  "%",    20.0,    3),
        ("input-radiation",           "Shortwave Radiation",          "W/m²", 500.0,   3),
        ("input-pressure",            "Surface Pressure",             "hPa",  880.0,   3),
        ("input-dew-point",           "Dew Point (2m)",               "°C",   12.0,    3),
    ]),
]

RISK_CLASS = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}


def build_input(input_id, label, unit, default, width):
    return dbc.Col(
        html.Div([
            html.Div([
                html.Label(label, htmlFor=input_id, className="input-label"),
                html.Span(unit, className="input-unit"),
            ], className="input-label-row"),
            dcc.Input(id=input_id, type="number", value=default,
                      className="form-control pro-input"),
        ], className="input-wrapper"),
        md=width, sm=6, xs=12, className="mb-3",
    )


def build_group_header(text):
    return html.Div(
        [html.Span(className="group-bar"), html.Span(text, className="group-text")],
        className="group-label",
    )


def build_layout():
    groups = []
    for name, specs in INPUT_GROUPS:
        groups.append(build_group_header(name))
        groups.append(dbc.Row([build_input(*s) for s in specs], className="g-3 mb-2"))

    return dbc.Container([
        html.Div([
            html.Div([
                html.Span(className="pulse-dot"),
                html.Span("Model Online", className="status-text"),
            ], className="status-badge"),
        ], className="header-top"),
        html.H1("Load-Shedding Risk Prediction", className="app-title"),
        html.P("Eskom System & Weather Risk Model — Capstone Dashboard",
               className="app-subtitle"),

        dbc.Card(dbc.CardBody([
            html.H5("Input Parameters", className="section-title mb-1"),
            html.P("Adjust the current grid and weather readings, then run the model.",
                   className="section-subtitle mb-3"),
            *groups,
        ]), className="pro-card mt-4"),

        html.Div(
            html.Button("Run Prediction", id="predict-button", n_clicks=0,
                        className="pro-btn"),
            className="d-flex justify-content-center my-4",
        ),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Predicted Probability", className="metric-label"),
                html.Div("--", id="probability-output", className="metric-value"),
            ]), className="pro-card h-100"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Div("Risk Level", className="metric-label"),
                html.Div("--", id="risk-level-output", className="metric-value"),
            ]), className="pro-card h-100"), md=6),
        ], className="g-3 mb-3"),

       dbc.Card(dbc.CardBody(
    dcc.Graph(id="gauge-chart",
              figure=build_gauge(0.0, THRESHOLD),
              config={"displayModeBar": False},
              className="chart-clean"),
            ), className="pro-card mb-4"),

        html.Footer(
            html.Div([
                html.Span("Load-Shedding Risk Prediction Capstone", className="footer-strong"),
                html.Span(" · ", className="footer-sep"),
                html.Span(f"Decision threshold: {THRESHOLD:.4f}", className="footer-muted"),
            ], className="footer-inner"),
            className="app-footer",
        ),
    ], fluid=True, className="px-4 pb-5")


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------
def build_gauge(prob, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 44, "color": "#e6edf7"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b98ad",
                     "tickfont": {"color": "#8b98ad"}},
            "bar": {"color": "#f59e0b", "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(16,185,129,0.18)"},
                {"range": [threshold * 100, 50], "color": "rgba(245,158,11,0.18)"},
                {"range": [50, 100], "color": "rgba(239,68,68,0.18)"},
            ],
            "threshold": {
                "line": {"color": "#ef4444", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#e6edf7"},
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        height=340,
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Load-Shedding Risk Prediction"
app.layout = build_layout()



@app.callback(
    Output("probability-output", "children"),
    Output("risk-level-output", "children"),
    Output("risk-level-output", "className"),
    Output("gauge-chart", "figure"),
    Input("predict-button", "n_clicks"),
    State("input-residual-demand", "value"),
    State("input-rsa-contracted", "value"),
    State("input-dispatchable-gen", "value"),
    State("input-thermal-gen", "value"),
    State("input-nuclear-gen", "value"),
    State("input-total-re", "value"),
    State("input-temperature", "value"),
    State("input-wind-speed", "value"),
    State("input-cloud-cover", "value"),
    State("input-radiation", "value"),
    State("input-pressure", "value"),
    State("input-dew-point", "value"),
    prevent_initial_call=True,
)
def on_predict(_n, demand, rsa_fc, disp_gen, thermal, nuclear, re,
               temp, wind, cloud, rad, pressure, dew):
    raw = {
        "residual_demand":        demand,
        "rsa_contracted_forecast": rsa_fc,
        "dispatchable_generation": disp_gen,
        "thermal_generation":     thermal,
        "nuclear_generation":     nuclear,
        "total_re":               re,
        "temperature_2m":         temp,
        "wind_speed_10m":         wind,
        "cloud_cover":            cloud,
        "shortwave_radiation":    rad,
        "surface_pressure":       pressure,
        "dew_point_2m":           dew,
    }
    raw = {k: (0.0 if v is None else float(v)) for k, v in raw.items()}

    prob, risk = predict(raw)
    cls = f"metric-value {RISK_CLASS[risk]}"
    return f"{prob:.1%}", risk, cls, build_gauge(prob, THRESHOLD)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}{%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg:#0a0f1c; --surface:#0f1729; --surface-2:#131d33; --surface-3:#182238;
            --border:#1e2b45; --border-strong:#2a3a55;
            --text:#e6edf7; --text-muted:#8b98ad; --text-dim:#5f6d85;
            --accent:#f59e0b; --accent-2:#fb923c;
            --emerald:#10b981; --danger:#ef4444;
            --radius:14px; --radius-sm:10px;
            --shadow-md:0 8px 24px rgba(0,0,0,0.35);
            --shadow-glow:0 0 0 3px rgba(245,158,11,0.15);
        }
        html, body { background: var(--bg) !important; color: var(--text);
            font-family: 'Inter', -apple-system, sans-serif; -webkit-font-smoothing: antialiased; }
        body::before {
            content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
            background:
                radial-gradient(1200px 600px at 15% -10%, rgba(245,158,11,0.06), transparent 60%),
                radial-gradient(900px 500px at 100% 0%, rgba(56,189,248,0.05), transparent 55%);
        }
        .container-fluid { position:relative; z-index:1; max-width:1400px; }

        .app-title { font-size:2rem; font-weight:700; letter-spacing:-0.02em; margin:0;
            background:linear-gradient(180deg,#fff 0%,#cbd5e1 100%);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .app-subtitle { color:var(--text-muted); margin:6px 0 0; font-size:0.95rem; }
        .header-top { display:flex; justify-content:flex-end; padding-top:24px; }
        .status-badge { display:inline-flex; align-items:center; gap:8px;
            padding:5px 12px; border-radius:999px;
            background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25);
            font-size:12px; color:#6ee7b7; font-weight:500; }
        .pulse-dot { width:7px; height:7px; border-radius:50%; background:#10b981;
            animation:pulse 2s infinite; }
        @keyframes pulse {
            0%{box-shadow:0 0 0 0 rgba(16,185,129,0.55);}
            70%{box-shadow:0 0 0 8px rgba(16,185,129,0);}
            100%{box-shadow:0 0 0 0 rgba(16,185,129,0);}
        }

        .pro-card { background:linear-gradient(180deg,var(--surface) 0%,var(--surface-2) 100%) !important;
            border:1px solid var(--border) !important; border-radius:var(--radius) !important;
            box-shadow:var(--shadow-md); transition:border-color .2s, transform .2s; }
        .pro-card:hover { border-color:var(--border-strong) !important; transform:translateY(-1px); }
        .pro-card .card-body { padding:22px 24px; }

        .section-title { font-weight:600; color:var(--text); margin:0; }
        .section-subtitle { color:var(--text-muted); font-size:0.875rem; }

        .group-label { display:flex; align-items:center; gap:10px; margin:18px 0 12px; }
        .group-bar { width:3px; height:14px; border-radius:2px;
            background:linear-gradient(180deg,var(--accent),var(--accent-2)); }
        .group-text { font-size:0.75rem; font-weight:600; letter-spacing:0.12em;
            text-transform:uppercase; color:var(--text-muted); }

        .input-wrapper { display:flex; flex-direction:column; }
        .input-label-row { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }
        .input-label { font-size:0.8rem; font-weight:500; color:var(--text-muted); margin:0; }
        .input-unit { font-size:0.7rem; color:var(--text-dim); font-weight:500;
            padding:2px 6px; border-radius:4px; background:rgba(148,163,184,0.06);
            border:1px solid rgba(148,163,184,0.08); }

        .form-control, .pro-input {
            background:var(--surface-3) !important; border:1px solid var(--border) !important;
            color:var(--text) !important; border-radius:var(--radius-sm) !important;
            padding:10px 14px !important; font-size:0.95rem; font-weight:500;
            font-variant-numeric:tabular-nums; transition:border-color .18s, box-shadow .18s; }
        .form-control:hover { border-color:var(--border-strong) !important; }
        .form-control:focus, .pro-input:focus {
            border-color:var(--accent) !important; box-shadow:var(--shadow-glow) !important;
            outline:none !important; }

        .pro-btn { background:linear-gradient(135deg,#f59e0b 0%,#f97316 100%);
            color:#0a0f1c; border:none; border-radius:var(--radius-sm);
            padding:14px 42px; font-weight:600; font-size:0.95rem;
            letter-spacing:0.03em; text-transform:uppercase;
            box-shadow:0 8px 24px rgba(245,158,11,0.28);
            transition:transform .15s, box-shadow .2s, filter .2s; cursor:pointer; }
        .pro-btn:hover { transform:translateY(-2px);
            box-shadow:0 12px 32px rgba(245,158,11,0.42); filter:brightness(1.05); }
        .pro-btn:active { transform:translateY(0); }

        .metric-label { font-size:0.75rem; font-weight:600; letter-spacing:0.12em;
            text-transform:uppercase; color:var(--text-muted); margin-bottom:10px; text-align:center; }
        .metric-value { font-size:2.25rem; font-weight:700; letter-spacing:-0.02em;
            color:var(--text); font-variant-numeric:tabular-nums;
            line-height:1.1; text-align:center; }

        .risk-low    { color:var(--emerald) !important; text-shadow:0 0 24px rgba(16,185,129,0.35); }
        .risk-medium { color:var(--accent)  !important; text-shadow:0 0 24px rgba(245,158,11,0.35); }
        .risk-high   { color:var(--danger)  !important; text-shadow:0 0 24px rgba(239,68,68,0.35); }

        .app-footer { margin-top:40px; border-top:1px solid var(--border); padding-top:20px; }
        .footer-inner { text-align:center; font-size:0.8rem; color:var(--text-dim); }
        .footer-strong { color:var(--text-muted); font-weight:500; }

        ::-webkit-scrollbar { width:10px; height:10px; }
        ::-webkit-scrollbar-track { background:var(--bg); }
        ::-webkit-scrollbar-thumb { background:var(--surface-3); border-radius:6px; border:2px solid var(--bg); }
        ::-webkit-scrollbar-thumb:hover { background:var(--border-strong); }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""


if __name__ == "__main__":
    import os
    app.run(host="127.0.0.1", port=8050, debug=False)