import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_antd_components as sac
import toml
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from streamlit_extras.mandatory_date_range import date_range_picker
from utils.styles import styling

st.set_page_config(layout="wide")
st.title("Utilization")

# Simplified CSS for layout
st.markdown(
    """
    <style>
    [data-testid="collapsedControl"] {display: none;}
    #MainMenu, header, footer {visibility: hidden;}
    .appview-container .main .block-container {
        padding: 1px 1rem 1rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Configuration and Database Setup
def load_config():
    try:
        with open(".streamlit/secrets.toml") as f:
            return OmegaConf.create(toml.load(f))
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

def create_session(cfg):
    if not cfg:
        return None, None
    try:
        db = cfg.connections.postgresql
        engine_url = f"{db.dialect}://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"
        engine = create_engine(engine_url)
        return sessionmaker(bind=engine)(), engine
    except Exception as e:
        st.error(f"Error creating database session: {e}")
        return None, None

# Helper Functions
def determine_sector(cell: str) -> int:
    sector_mapping = {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2, "6": 3, "7": 1, "8": 2, "9": 3}
    return sector_mapping.get(cell[-1].upper() if cell else "0", 0)

def colors():
    return ["#9e0142", "#fdae61", "#66c2a5", "#d53e4f", "#5e4fa2", "#f46d43", "#abdda4", "#3288bd", "#fee08b", "#e6f598"]

def get_header(cells: list[str]) -> str:
    sectors = sorted({f"Sector {cell[-1]}" for cell in cells if cell})
    return ", ".join(sectors) or "No Sectors"

def create_chart(df, site, parameter):
    if df.empty:
        st.warning(f"No data available for {parameter} at site {site}.")
        return go.Figure()

    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S.%f")
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    title = get_header(df["Cell Name"].unique())
    color_mapping = dict(zip(df["Cell Name"].unique(), colors()))

    for _, sector_df in df.groupby("sector"):
        for cell, cell_df in sector_df.groupby("Cell Name"):
            cell_df_sorted = cell_df.sort_values("Time")
            if parameter in cell_df_sorted.columns and not cell_df_sorted.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cell_df_sorted["Time"],
                        y=cell_df_sorted[parameter],
                        mode="lines",
                        name=cell,
                        line=dict(color=color_mapping.get(cell, "#000000")),
                        hovertemplate=f"<b>{cell}</b><br><b>Date:</b> %{{x}}<br><b>{parameter}:</b> %{{y}}<br><extra></extra>",
                    )
                )

    fig.update_layout(
        title_text=title if fig.data else f"No data available for {title}",
        title_x=0.4,
        template="plotly_white",
        # xaxis=dict(tickformat="%m/%d/%Y %H:%M", tickangle=-45, type="date", tickmode="auto", nticks=15),
        xaxis=dict(tickformat="%m/%d/%Y", tickangle=-45, type="date", tickmode="auto", nticks=15),
        yaxis_title=parameter,
        autosize=True,
        height=400,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.4,
            xanchor="center",
            x=0.5,
            bgcolor="#F5F5F5",
            bordercolor="#F5F5F5",
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            itemsizing="constant",
            font=dict(size=14),
        ),
    )
    return fig

# Main Function
def main():
    cfg = load_config()
    session_key = "db_session"
    if session_key not in st.session_state:
        st.session_state[session_key], st.session_state["engine"] = create_session(cfg)
    session, engine = st.session_state[session_key], st.session_state.get("engine")

    if not engine:
        return

    try:
        with open(os.path.join(os.path.dirname(__file__), "sitelist.txt")) as f:
            site_list = [line.strip() for line in f]
    except FileNotFoundError as e:
        st.error(f"Error loading site list: {e}")
        return

    selected_sites = st.multiselect("Select Site IDs", site_list)

    if "date_range" not in st.session_state:
        st.session_state["date_range"] = (pd.Timestamp.today() - pd.Timedelta(days=14), pd.Timestamp.today())

    date_range = date_range_picker("DATE RANGE", *st.session_state["date_range"])
    df = None

    # Combine the two st.button calls into one
    if st.button("Run Query"):
        if not selected_sites or not date_range:
            st.warning("Please select site IDs and date range to load data.")
            return
        try:
            start_date, end_date = date_range
            like_conditions = " OR ".join([f'"eNodeB Name" LIKE :site_{i}' for i in range(len(selected_sites))])
            query = text(
                """
                SELECT
                    "Time", "eNodeB Name", "Cell Name",
                    "DL Resource Block Utilizing Rate %_FIX",
                    "Active User", "User Downlink Average Throughput (Mbps)"
                FROM ltehourly
                WHERE ({}) AND "Time" BETWEEN :start_date AND :end_date
                """.format(like_conditions)
            )
            params = {f"site_{i}": f"%{site}%" for i, site in enumerate(selected_sites)}
            params.update({"start_date": start_date, "end_date": end_date})

            df = pd.read_sql(query, engine, params=params)
            df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S.%f")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    if df is not None and not df.empty:
        df["site"] = df["eNodeB Name"]
        df["sector"] = df["Cell Name"].apply(determine_sector)

        for site in selected_sites:
            site_df = df[df["eNodeB Name"].str.contains(site, na=False)]
            if site_df.empty:
                st.warning(f"No data available for site {site}.")
                continue

            sac.divider(label=site, icon="graph-up", align="center", size="xl", color="indigo")
            parameters = [
                ("Utilization", "DL Resource Block Utilizing Rate %_FIX"),
                ("Active User", "Active User"),
                ("User Throughput", "User Downlink Average Throughput (Mbps)"),
            ]

            for title, param in parameters:
                st.markdown(
                    *styling(
                        f"📶 {title} {site}",
                        font_size=24,
                        text_align="left",
                        tag="h6",
                        font_color="#D90013",
                    )
                )
                cols = st.columns(3)
                for sector, col in zip(sorted(site_df["sector"].unique()), cols):
                    with col.container(border=True):
                        sector_df = site_df[site_df["sector"] == sector]
                        fig = create_chart(sector_df, site, param)
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()