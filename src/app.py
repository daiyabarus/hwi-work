import os

import pandas as pd
import streamlit as st

from charts import ChartGenerator
from config import ConfigLoader
from constants import ASSETS_DIR
from database import DatabaseSession, QueryManager
from ui import StreamlitUI


class App:
    def __init__(self):
        self.config = ConfigLoader.load()
        self.db_session = DatabaseSession(self.config)
        self.ui = StreamlitUI()
        self.charts = ChartGenerator()
        self.df_manager = {}  # Simplified DataFrame manager

    def run(self):
        with self.db_session as session:
            if not session:
                return
            query_manager = QueryManager(self.db_session.get_engine())

            # Load sitelist
            script_dir = os.path.dirname(__file__)
            sitelist = self.ui.load_sitelist(
                os.path.join(script_dir, "test_sitelist.csv")
            )

            # UI Components
            col1, col2, col3, col4, _ = st.columns([1, 1, 1, 1, 3])
            with col1:
                date_range = self.ui.select_date_range()
                st.session_state["date_range"] = date_range
            with col2:
                selected_sites = self.ui.select_options(sitelist, "SITEID", 0)
                st.session_state["selected_sites"] = selected_sites
            with col3:
                selected_neids = self.ui.select_options(sitelist, "NEID", 2)
                st.session_state["selected_neids"] = selected_neids
            with col4:
                xrule = self.ui.select_xrule_date()
                st.session_state["xrule"] = xrule

            # Run Query
            if st.button("Run Query") and selected_sites and date_range:
                start_date, end_date = date_range
                for siteid in selected_sites:
                    # Fetch and store data
                    ltedaily_data = query_manager.get_ltedaily_data(
                        siteid, selected_neids, start_date, end_date
                    )
                    self.df_manager[f"ltedaily_{siteid}"] = ltedaily_data
                    vswr_data = query_manager.get_vswr_data(selected_sites, end_date)
                    self.df_manager[f"vswr_{siteid}"] = vswr_data

                    # Display charts
                    st.header(f"Data for Site {siteid}")
                    self.charts.create_line_chart(
                        ltedaily_data,
                        "DATE_ID",
                        "Availability",
                        "EutranCell",
                        f"Availability for {siteid}",
                        xrule=True,
                    )
                    self.charts.create_vswr_chart(
                        vswr_data, "DATE_ID", "RRU", "VSWR", f"VSWR for {siteid}"
                    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="collapsedControl"] {display: none;}
        #MainMenu, header, footer {visibility: hidden;}
        .appview-container .main .block-container {
            padding-top: 1px; padding-left: 1rem; padding-right: 1rem; padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    App().run()
