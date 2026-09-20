import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Bronx Heat Twin Game", layout="wide")

DATA_FILE = r"bronx_heat_twin_master.geojson"
DEFAULT_BETA = 6.0


# =========================
# HELPERS
# =========================
def norm(series, vmin=None, vmax=None):
    """Min-max normalize to 0-1. Pass fixed vmin/vmax to keep the scale
    stable across reruns instead of recomputing it from whatever subset
    of rows happens to be in `series` at the time."""
    s = series.astype(float)
    if vmin is None:
        vmin = s.min()
    if vmax is None:
        vmax = s.max()
    return (s - vmin) / (vmax - vmin)


@st.cache_data
def load_data():
    gdf = gpd.read_file(DATA_FILE)
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    gdf = gdf.to_crs(epsg=4326)
    # Reset index so it's a clean 0..N-1 range that matches `locations=gdf.index`
    gdf = gdf.reset_index(drop=True)
    return gdf


def apply_scenario(gdf, selected_geoids, canopy_add, beta, temp_min, temp_max):
    gdf = gdf.copy()

    gdf["canopy_scenario"] = gdf["canopy_mean"]
    gdf["temp_scenario"] = gdf["temp_mean"]

    if selected_geoids:
        mask = gdf["GEOID"].isin(selected_geoids)

        # Increase canopy. canopy_mean is a 0-1 FRACTION (e.g. 0.47 = 47%
        # canopy cover). This ADDS a flat amount (canopy_add, e.g. 0.10 =
        # "+10 percentage points") to every selected tract, regardless of
        # its starting canopy - so a tract at 3% canopy and a tract at 47%
        # canopy get the same absolute tree-planting benefit. (The old
        # multiplicative version - canopy_mean * (1 + pct) - made Δcanopy
        # proportional to existing canopy, which rewarded already-green
        # tracts and gave the lowest-canopy, often highest-vulnerability
        # tracts the smallest benefit for the same policy effort.)
        gdf.loc[mask, "canopy_scenario"] = (
            gdf.loc[mask, "canopy_mean"] + canopy_add
        ).clip(0, 1)

        # Cooling effect. delta_canopy is already a 0-1 fraction change,
        # so no extra /100 here (that was double-dividing an already-small
        # fractional value, shrinking the whole effect ~100x).
        delta_canopy = (
            gdf.loc[mask, "canopy_scenario"] - gdf.loc[mask, "canopy_mean"]
        )

        gdf.loc[mask, "temp_scenario"] = (
            gdf.loc[mask, "temp_mean"] - beta * delta_canopy
        )

    # Renormalize temperature using FIXED baseline bounds, not bounds
    # recomputed from this call's data. Otherwise a cooled tract can
    # itself shift the min/max, which quietly distorts the scale run
    # to run and makes the true magnitude of a scenario hard to read.
    gdf["T_scenario"] = norm(gdf["temp_scenario"], vmin=temp_min, vmax=temp_max)

    # Scenario risk
    gdf["RISK_SCENARIO"] = (
        gdf["T_scenario"] + gdf["V"] - gdf["C"]
    ).clip(0, 1)

    gdf["DELTA_RISK"] = gdf["RISK_INDEX"] - gdf["RISK_SCENARIO"]

    return gdf


def make_plotly_map(gdf, value_col, selected_geoids):
    fig = px.choropleth(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color=value_col,
        color_continuous_scale="Reds",
        hover_data=["GEOID", "RISK_INDEX", "RISK_SCENARIO", "DELTA_RISK"],
    )

    fig.update_geos(fitbounds="locations", visible=False)

    # Highlight selected tracts with a blue outline overlay
    if selected_geoids:
        sel = gdf[gdf["GEOID"].isin(selected_geoids)]
        if not sel.empty:
            highlight = px.choropleth(
                sel,
                geojson=sel.geometry,
                locations=sel.index,
                color_discrete_sequence=["rgba(0,0,0,0)"],  # transparent fill
            ).data[0]
            highlight.marker.line.color = "blue"
            highlight.marker.line.width = 3
            highlight.showlegend = False
            highlight.hoverinfo = "skip"
            fig.add_trace(highlight)

    fig.update_layout(height=650, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def handle_selection(event, gdf):
    """Read the Plotly selection event and toggle GEOIDs into session state."""
    if not event or "selection" not in event:
        return

    points = event["selection"].get("points", [])
    if not points:
        return

    # Only look at points from the base choropleth trace (curveNumber 0),
    # so re-clicking the blue highlight overlay doesn't cause weirdness.
    clicked_indices = [
        p["location"] for p in points if p.get("curveNumber", 0) == 0
    ]
    if not clicked_indices:
        return

    clicked_geoids = gdf.loc[[int(i) for i in clicked_indices], "GEOID"].tolist()

    # Toggle: if already selected, remove; otherwise add.
    current = set(st.session_state.selected)
    for g in clicked_geoids:
        if g in current:
            current.discard(g)
        else:
            current.add(g)
    st.session_state.selected = list(current)


# =========================
# SESSION STATE
# =========================
if "selected" not in st.session_state:
    st.session_state.selected = []

if "scenario_applied" not in st.session_state:
    st.session_state.scenario_applied = False


# =========================
# LOAD DATA
# =========================
gdf_base = load_data()

# Fixed normalization bounds, computed once from the baseline dataset.
# Used for every scenario re-normalization so the 0-1 temperature scale
# never silently drifts between reruns/selections.
TEMP_MIN = float(gdf_base["temp_mean"].min())
TEMP_MAX = float(gdf_base["temp_mean"].max())

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Scenario Controls")

canopy_add = st.sidebar.slider(
    "Add canopy cover (percentage points, flat for every selected tract)",
    0, 50, 10, step=1,
) / 100
beta = st.sidebar.number_input(
    "Cooling sensitivity β (°C per +100% canopy)",
    value=float(DEFAULT_BETA),
    min_value=0.0,
    max_value=20.0,
)

layer = st.sidebar.radio(
    "Map layer",
    ["Baseline Risk", "Scenario Risk", "Risk Reduction (Delta)"],
)

apply_btn = st.sidebar.button("Apply Scenario", type="primary")
reset_btn = st.sidebar.button("Reset Selection")

st.sidebar.markdown(f"**Selected tracts:** {len(st.session_state.selected)}")

# Reset
if reset_btn:
    st.session_state.selected = []
    st.session_state.scenario_applied = False

# Apply scenario
if apply_btn:
    st.session_state.scenario_applied = True


# =========================
# APPLY SCENARIO IF ACTIVE
# =========================
if st.session_state.scenario_applied:
    gdf = apply_scenario(
        gdf_base,
        st.session_state.selected,
        canopy_add,
        beta,
        TEMP_MIN,
        TEMP_MAX,
    )
else:
    gdf = gdf_base.copy()
    gdf["RISK_SCENARIO"] = gdf["RISK_INDEX"]
    gdf["DELTA_RISK"] = 0
    gdf["canopy_scenario"] = gdf["canopy_mean"]
    gdf["temp_scenario"] = gdf["temp_mean"]


# Choose layer column
if layer == "Baseline Risk":
    col = "RISK_INDEX"
elif layer == "Scenario Risk":
    col = "RISK_SCENARIO"
else:
    col = "DELTA_RISK"


# =========================
# MAIN LAYOUT
# =========================
st.title("🔥 Bronx Heat Digital Twin Game")

st.markdown(
    "### Click a tract to select it (shift-click for more), "
    "or use the box/lasso tool in the chart toolbar for multi-select."
)

# Plotly interactive map with native Streamlit selection support
fig = make_plotly_map(gdf, col, st.session_state.selected)

event = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode=["points", "box", "lasso"],
    key="map",
)

handle_selection(event, gdf)

# =========================
# TABLE BELOW MAP
# =========================
st.subheader("Selected tracts (Before vs After)")

if st.session_state.selected:
    sel = gdf[gdf["GEOID"].isin(st.session_state.selected)]

    table = sel[
        [
            "GEOID",
            "RISK_INDEX",
            "RISK_SCENARIO",
            "DELTA_RISK",
            "canopy_mean",
            "canopy_scenario",
            "temp_mean",
            "temp_scenario",
        ]
    ].sort_values("DELTA_RISK", ascending=False)

    st.dataframe(table, use_container_width=True)

else:
    st.warning("No tracts selected yet. Click on the map.")

# =========================
# DASHBOARD METRICS
# =========================
st.subheader("Scenario Impact Summary")

baseline_mean = gdf["RISK_INDEX"].mean()
scenario_mean = gdf["RISK_SCENARIO"].mean()

# Citywide numbers are averaged over ALL ~355 Bronx tracts. Only the tracts
# you selected actually change, so a few tracts' worth of canopy will barely
# move this average - that's realistic, not a bug, but it needs more decimal
# places to actually be visible instead of rounding to 0.000.
st.metric("Citywide Baseline Mean Risk", f"{baseline_mean:.3f}")
st.metric("Citywide Scenario Mean Risk", f"{scenario_mean:.4f}")
st.metric("Citywide Mean Risk Reduction", f"{(baseline_mean - scenario_mean):.5f}")

# Selected-tracts numbers isolate the impact to just what you acted on, so
# they move visibly whenever you change the selection or the sliders.
if st.session_state.selected:
    sel = gdf[gdf["GEOID"].isin(st.session_state.selected)]
    st.metric("Selected Tracts: Mean Risk Reduction", f"{sel['DELTA_RISK'].mean():.3f}")
    st.metric("Selected Tracts: Total Risk Reduction", f"{sel['DELTA_RISK'].sum():.3f}")
