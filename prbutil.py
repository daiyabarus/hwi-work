import os
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
import toml
from omegaconf import OmegaConf
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from streamlit_extras.mandatory_date_range import date_range_picker

st.set_page_config(layout="wide")


def load_config():
    try:
        with open(".streamlit/secrets.toml") as f:
            return OmegaConf.create(toml.load(f))
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None


def create_session(cfg):
    if cfg is None:
        return None, None
    try:
        db_cfg = cfg.connections.postgresql
        engine_url = f"{db_cfg.dialect}://{db_cfg.username}:{db_cfg.password}@{db_cfg.host}:{db_cfg.port}/{db_cfg.database}"
        engine = create_engine(engine_url)
        Session = sessionmaker(bind=engine)
        return Session(), engine
    except Exception as e:
        st.error(f"Error creating database session: {e}")
        return None, None


def group_sector(sector: str) -> int:
    sector_mapping = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 1,
        "5": 2,
        "6": 3,
        "7": 1,
        "8": 2,
        "9": 3,
        "10": 1,
        "11": 2,
        "12": 3,
    }
    return sector_mapping.get(sector, 0)


def run_query(
    engine: Engine,
    selected_sites: list[str],
    date_range: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame | None:
    if not selected_sites or not date_range:
        st.warning("Please select site IDs and date range to load data.")
        return None
    start_date, end_date = date_range
    like_conditions = " OR ".join(
        [f'"siteid" LIKE :site_{i}' for i in range(len(selected_sites))]
    )
    query = text(f"""
        SELECT "date", "time", "siteid", "nename", "Band Type", "cellname", "Sector",
               "DL_Resource_Block_Utilizing_Rate(%)", "New Active User", "User_DL_Avg_Throughput(Kbps)"
        FROM ltehourly
        WHERE ({like_conditions}) AND "date" BETWEEN :start_date AND :end_date
    """)
    params = {f"site_{i}": f"%{site}%" for i, site in enumerate(selected_sites)}
    params.update({"start_date": start_date, "end_date": end_date})
    return pd.read_sql(query, engine, params=params)


def main():
    st.title("PRB Utilization Data Viewer")

    cfg = load_config()
    session, engine = (
        create_session(cfg)
        if "db_session" not in st.session_state
        else (st.session_state.db_session, st.session_state.engine)
    )
    st.session_state.db_session, st.session_state.engine = session, engine

    if engine:
        try:
            with open(os.path.join(os.path.dirname(__file__), "sitelist.txt")) as f:
                site_list = [line.strip() for line in f]
            selected_sites = st.multiselect("Select Site IDs", options=site_list)

            if "date_range" not in st.session_state:
                st.session_state["date_range"] = (
                    pd.Timestamp.today() - pd.Timedelta(days=14),
                    pd.Timestamp.today(),
                )

            date_range = date_range_picker(
                "DATE RANGE",
                default_start=st.session_state["date_range"][0],
                default_end=st.session_state["date_range"][1],
            )

            if st.button("Run Query"):
                df = run_query(engine, selected_sites, date_range)
                if df is not None:
                    st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading data: {e}")


if __name__ == "__main__":
    main()
