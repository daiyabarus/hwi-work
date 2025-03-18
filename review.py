import os
from typing import Any

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import toml
from omegaconf import DictConfig, OmegaConf
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from streamlit_extras.mandatory_date_range import date_range_picker
from streamlit_extras.stylable_container import stylable_container

from colors import ColorPalette
from styles import styling

pd.options.mode.copy_on_write = True


class Config:
    def load(self):
        with open(".streamlit/secrets.toml") as f:
            cfg = OmegaConf.create(toml.loads(f.read()))
        return cfg


class DatabaseSession:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def create_session(self):
        try:
            db_cfg = self.cfg.connections.postgresql
            engine = create_engine(
                f"{db_cfg.dialect}://{db_cfg.username}:{db_cfg.password}@{db_cfg.host}:{db_cfg.port}/{db_cfg.database}"
            )
            Session = sessionmaker(bind=engine)
            return Session(), engine
        except Exception as e:
            st.error(f"Error creating database session: {e}")
            return None, None


class DataFrameManager:
    def __init__(self):
        self.dataframes = {}

    def add_dataframe(self, name, dataframe):
        self.dataframes[name] = dataframe

    def get_dataframe(self, name):
        return self.dataframes.get(name)

    def display_dataframe(self, name, header):
        dataframe = self.get_dataframe(name)
        if dataframe is not None:
            st.header(header)
            st.write(dataframe)


class StreamlitInterface:
    def load_sitelist(self, filepath):
        with open(filepath) as file:
            data = [line.strip() for line in file if line.strip()]
        return data

    def site_selection(self, sitelist):
        return st.multiselect("SITEID", sitelist)

    def neid_selection(self):
        band_options = ["L1800", "L900", "L2100", "L2300-ME", "L2300-MF", "L2300-MV"]
        return st.multiselect("BAND", band_options)

    def select_date_range(self):
        if "date_range" not in st.session_state:
            st.session_state["date_range"] = (
                pd.Timestamp.today() - pd.DateOffset(days=15),
                pd.Timestamp.today(),
            )

        date_range = date_range_picker(
            "DATE RANGE",
            default_start=st.session_state["date_range"][0],
            default_end=st.session_state["date_range"][1],
        )
        return date_range


class QueryManager:
    def __init__(self, engine):
        self.engine = engine

    def _fetch_data(
        self, query: str, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        try:
            df = pd.read_sql(query, self.engine, params=params)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def _build_like_conditions(
        self, field: str, values: list[str]
    ) -> tuple[str, dict[str, str]]:
        conditions = " OR ".join(
            [f'"{field}" LIKE :param_{i}' for i in range(len(values))]
        )
        params = {f"param_{i}": f"%{value}%" for i, value in enumerate(values)}
        return conditions, params

    @st.cache_data(ttl=600)
    def get_ltedaily_data(
        _self,
        siteid: str,
        start_date: str,
        end_date: str,
        band: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch LTE daily data with optional multiple band filter."""
        base_query = """
            SELECT * FROM ltedaily
            WHERE "siteid" LIKE :siteid
            AND "date" BETWEEN :start_date AND :end_date
        """
        params = {"siteid": siteid, "start_date": start_date, "end_date": end_date}

        if band:
            if len(band) == 1:
                base_query += ' AND "band" LIKE :band'
                params["band"] = band[0]
            else:
                band_placeholders = ", ".join([f":band_{i}" for i in range(len(band))])
                base_query += f' AND "band" IN ({band_placeholders})'
                for i, b in enumerate(band):
                    params[f"band_{i}"] = b

        query = text(base_query)
        return _self._fetch_data(query, params)

    @st.cache_data(ttl=600)
    def get_ltedaily_wo_band(
        _self, siteid: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        base_query = """
            SELECT * FROM ltedaily
            WHERE "siteid" LIKE :siteid
            AND "date" BETWEEN :start_date AND :end_date
        """
        params = {"siteid": siteid, "start_date": start_date, "end_date": end_date}
        query = text(base_query)
        return _self._fetch_data(query, params)

    @st.cache_data(ttl=600)
    def get_ltehourly_data(_self, siteids: list[str]) -> pd.DataFrame:
        conditions, params = _self._build_like_conditions("eNodeB Name", siteids)
        query = text(f"""
            SELECT *
            FROM ltehourly
            WHERE ({conditions})
        """)
        df = _self._fetch_data(query, params)
        if not df.empty:
            df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S.%f")
        return df

    @st.cache_data(ttl=600)
    def get_ltetastate_data(_self, siteids: list[str]) -> pd.DataFrame:
        conditions, params = _self._build_like_conditions("enodeb_name", siteids)
        query = text(f"""
            SELECT * FROM timingadvance
            WHERE ({conditions})
        """)
        return _self._fetch_data(query, params)


class ChartGenerator:
    def __init__(self):
        self.color_palette = ColorPalette()

    def get_colors(self, num_colors):
        return [self.color_palette.get_color(i) for i in range(num_colors)]

    def get_header(self, cell, param, site):
        result = []
        for input_string in cell:
            last_char = input_string[-1]
            sector = self.determine_sector(last_char)
            formatted_string = f"Sector {sector}"
            if formatted_string not in result:
                result.append(formatted_string)
        result.sort()
        return f"{param} {site}"

    def get_headers(self, cells):
        sectors = {f"Sector {input_string[-1]}" for input_string in cells}
        sorted_sectors = sorted(sectors)
        return ", ".join(sorted_sectors)

    def determine_sector(self, cell: str) -> int:
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
        }
        last_char = cell[-1].upper()
        return sector_mapping.get(last_char, 0)

    def create_charts_for_daily(
        self, df, cell_name, x_param, y_param, xrule=False, yline=None
    ):
        df = df.sort_values(by=x_param)
        df[y_param] = df[y_param].astype(float)
        df = df[df[y_param] != 0]
        df["sector"] = df[cell_name].apply(self.determine_sector)

        color_mapping = {
            cell: color
            for cell, color in zip(
                sorted(df[cell_name].unique()),
                self.get_colors(len(df[cell_name].unique())),
            )
        }

        sector_count = df["sector"].nunique()
        cols = max(min(sector_count, 3), 1)
        columns = st.columns(cols)

        for idx, sector in enumerate(sorted(df["sector"].unique())):
            sector_data = df[df["sector"] == sector]
            y_min = sector_data[sector_data[y_param] > 0][y_param].min()
            y_max_value = sector_data[y_param].max()
            y_max = (
                100
                if 95 < y_max_value <= 100
                else y_max_value
                if y_max_value > 100
                else y_max_value
            )

            with columns[idx % cols]:
                with stylable_container(
                    key=f"container_with_border_{sector}",
                    css_styles="""
                                {
                                    background-color: #F5F5F5;
                                    border: 2px solid rgba(49, 51, 63, 0.2);
                                    border-radius: 0.5rem;
                                    padding: calc(1em - 1px)
                                }
                                """,
                ):
                    container = st.container()
                    with container:
                        fig = go.Figure()

                        for cell in sector_data[cell_name].unique():
                            cell_data = sector_data[sector_data[cell_name] == cell]
                            color = color_mapping[cell]

                            fig.add_trace(
                                go.Scatter(
                                    x=cell_data[x_param],
                                    y=cell_data[y_param],
                                    mode="lines",
                                    name=cell,
                                    line=dict(color=color, width=2),
                                    hovertemplate=(
                                        f"<b>{cell}</b><br>"
                                        f"<b>{y_param}:</b> %{{y}}<br>"
                                        "<extra></extra>"
                                    ),
                                )
                            )

                        if yline is not None:
                            # Check if yline is a column name or a static value
                            if isinstance(yline, str) and yline in df.columns:
                                yline_value = sector_data[yline].mean()
                            else:
                                try:
                                    yline_value = float(
                                        yline
                                    )  # Convert static input to float
                                except (ValueError, TypeError):
                                    st.warning(
                                        f"Invalid yline value: {yline}. Skipping yline."
                                    )
                                    yline_value = None

                            if yline_value is not None:
                                fig.add_hline(
                                    y=yline_value,
                                    line_dash="dashdot",
                                    line_color="#F70000",
                                    line_width=2,
                                )
                                if yline_value > y_max:
                                    y_max = yline_value
                                elif yline_value < y_min:
                                    y_min = yline_value

                        if xrule:
                            fig.add_vline(
                                x=st.session_state["xrule"],
                                line_width=2,
                                line_dash="dash",
                                line_color="#808080",
                            )

                        adjusted_y_min = y_min if y_min > 0 else 0.01
                        yaxis_range = [adjusted_y_min, y_max]

                        fig.update_layout(
                            margin=dict(t=20, l=20, r=20, b=20),
                            title_text=f"SECTOR {sector}",
                            title_x=0.4,
                            template="plotly_white",
                            hoverlabel=dict(font_size=14, font_family="Vodafone"),
                            # hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.4,
                                xanchor="center",
                                x=0.5,
                                itemclick="toggleothers",
                                itemdoubleclick="toggle",
                                itemsizing="constant",
                                font=dict(size=14),
                                traceorder="normal",
                            ),
                            paper_bgcolor="#F5F5F5",
                            plot_bgcolor="#F5F5F5",
                            width=600,
                            height=250,
                            showlegend=True,
                            yaxis=dict(
                                range=yaxis_range,
                                tickfont=dict(
                                    size=14,
                                    color="#000000",
                                ),
                            ),
                            xaxis=dict(
                                tickfont=dict(
                                    size=14,
                                    color="#000000",
                                ),
                            ),
                        )

                        st.plotly_chart(fig, use_container_width=True)

    def create_daily_charts(self, df, site):
        parameters = [
            "rrc_setup_sr_service",
            "erab_setup_sr_all",
            "call_setup_sr",
            "service_drop_rate",
            "intrafreq_ho_out_sr",
            "inter_frequency_handover_sr",
            "lte_to_geran_redirection_sr",
            "csfb_execution_sr",
            "csfb_preparation_sr",
            "uplink_interference",
            "radio_network_availability_rate",
            "cqi",
            "se",
            "total_traffic_volume_gb",
            "total_user",
            "cell_dl_avg_throughput_mbps",
            "cell_ul_avg_throughput_mbps",
            "user_dl_avg_throughput_mbps",
            "user_ul_avg_throughput_mbps",
            "active_user",
            "ta_average",
        ]

        df["date"] = pd.to_datetime(df["date"])

        unique_cells = df["cell_name"].unique()
        # Generate colors for each unique cell
        colors = self.get_colors(len(unique_cells))
        color_mapping = dict(zip(unique_cells, colors))

        for i in range(0, len(parameters), 3):
            cols = st.columns(3, gap="small")
            containers = [
                cols[0].container(border=True),
                cols[1].container(border=True),
                cols[2].container(border=True),
            ]

            for j, param in enumerate(parameters[i : i + 3]):
                with containers[j]:
                    chart = (
                        alt.Chart(df)
                        .mark_line()
                        .encode(
                            x=alt.X("date:T", title=""),
                            y=alt.Y(f"{param}:Q", title=""),
                            color=alt.Color(
                                "cell_name:N",
                                scale=alt.Scale(
                                    domain=list(
                                        color_mapping.keys()
                                    ),  # List of cell names
                                    range=list(
                                        color_mapping.values()
                                    ),  # List of corresponding colors
                                ),
                                legend=alt.Legend(
                                    title=None,
                                    orient="bottom",
                                    columns=2,
                                    direction="horizontal",
                                    columnPadding=10,
                                    symbolLimit=0,
                                    labelLimit=0,
                                    labelFontSize=12,
                                    labelColor="black",
                                ),
                            ),
                            tooltip=["cell_name", "date", param],
                        )
                        .properties(
                            title=f"{param.replace('_', ' ').title().upper()} - {site}",
                            height=300,
                            width="container",
                        )
                        .interactive()
                    )

                    st.altair_chart(chart, use_container_width=True)


class App:
    def __init__(self):
        self.config = Config().load()
        self.database_session = DatabaseSession(self.config)
        self.query_manager = None
        self.dataframe_manager = DataFrameManager()
        self.streamlit_interface = StreamlitInterface()
        self.chart_generator = ChartGenerator()

    def run(self):
        session, engine = self.database_session.create_session()
        if session is None:
            return

        self.query_manager = QueryManager(engine)

        script_dir = os.path.dirname(__file__)
        sitelist_path = os.path.join(script_dir, "sitelist.txt")
        sitelist = self.streamlit_interface.load_sitelist(sitelist_path)

        with st.sidebar:
            date_range = self.streamlit_interface.select_date_range()
            st.session_state["date_range"] = date_range

            selected_sites = self.streamlit_interface.site_selection(sitelist)
            st.session_state.selected_sites = selected_sites

            selected_band = self.streamlit_interface.neid_selection()
            st.session_state.selected_band = selected_band

            if st.button("Run Query"):
                if selected_sites and date_range:
                    start_date, end_date = date_range
                    start_date_str = start_date.strftime("%Y-%m-%d")
                    end_date_str = end_date.strftime("%Y-%m-%d")

                    with st.spinner("Fetching data..."):
                        for site in selected_sites:
                            if selected_band:
                                df_daily = self.query_manager.get_ltedaily_data(
                                    site, start_date_str, end_date_str, selected_band
                                )
                                self.dataframe_manager.add_dataframe(
                                    f"dailysow_{site}", df_daily
                                )
                            else:
                                df_daily = self.query_manager.get_ltedaily_wo_band(
                                    site, start_date_str, end_date_str
                                )
                                self.dataframe_manager.add_dataframe(
                                    f"dailysow_{site}", df_daily
                                )

                            df_daily_all = self.query_manager.get_ltedaily_wo_band(
                                site, start_date_str, end_date_str
                            )
                            self.dataframe_manager.add_dataframe(
                                f"dailyall_{site}", df_daily_all
                            )

                            df_hourly = self.query_manager.get_ltehourly_data(
                                selected_sites
                            )
                            self.dataframe_manager.add_dataframe(
                                f"hourly_{site}", df_hourly
                            )

                            df_tastate = self.query_manager.get_ltetastate_data(
                                selected_sites
                            )
                            self.dataframe_manager.add_dataframe(
                                f"tastate_{site}", df_tastate
                            )

        if any(
            key.startswith("dailysow_") for key in self.dataframe_manager.dataframes
        ):
            tab1, tab2, tab3, tab4 = st.tabs(
                ["Sow Level", "Site Level", "Hourly", "TA"]
            )

            with tab1:
                for site in selected_sites:
                    df_daily_sow = self.dataframe_manager.get_dataframe(
                        f"dailysow_{site}"
                    )
                    if df_daily_sow is not None:
                        st.markdown(
                            *styling(
                                f"📶 RRC SR {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "rrc_setup_sr_service",
                            xrule=False,
                            yline="99.6",
                        )
                        st.markdown(
                            *styling(
                                f"📶 CSSR {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "call_setup_sr",
                            xrule=False,
                            yline="99.3",
                        )
                        st.markdown(
                            *styling(
                                f"📶 ERAB SR {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "erab_setup_sr_all",
                            xrule=False,
                            yline="99.7",
                        )
                        st.markdown(
                            *styling(
                                f"📶 SAR {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "service_drop_rate",
                            xrule=False,
                            yline="0.10",
                        )

            with tab2:
                for site in selected_sites:
                    df_daily_all = self.dataframe_manager.get_dataframe(
                        f"dailyall_{site}"
                    )
                    if df_daily_all is not None:
                        self.chart_generator.create_daily_charts(df_daily_all, site)

            with tab3:
                for site in selected_sites:
                    self.dataframe_manager.display_dataframe(
                        f"hourly_{site}", f"LTE Hourly Data - {site}"
                    )

            with tab4:
                for site in selected_sites:
                    self.dataframe_manager.display_dataframe(
                        f"tastate_{site}", f"LTE TA State Data - {site}"
                    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="collapsedControl"] {display: none;}
        #MainMenu, header, footer {visibility: hidden;}
        .appview-container .main .block-container {
            padding-top: 0.5px;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    app = App()
    app.run()
