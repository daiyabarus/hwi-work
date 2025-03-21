# import math
import os
from datetime import timedelta
from typing import Any, List, Optional

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_antd_components as sac
import toml
from omegaconf import DictConfig, OmegaConf

# from plotly.subplots import make_subplots
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
            st.dataframe(dataframe)


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

    def nbr_selection(self, sitelist):
        return st.multiselect("COSITE & 1ST TIER", sitelist)

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


class DateCalc:
    """A class for date-based calculations and aggregations on a DataFrame."""

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        date_column: str | None = None,
    ) -> None:
        """Initialize with optional DataFrame and date column.

        Args:
            dataframe: Input DataFrame (optional)
            date_column: Name of the date column (optional)
        """
        self._df: pd.DataFrame | None = None
        self._date_col: str | None = None
        if dataframe is not None and date_column is not None:
            self.set_dataframe(dataframe, date_column)

    def set_dataframe(self, dataframe: pd.DataFrame, date_column: str) -> None:
        """Set or update the DataFrame and date column.

        Args:
            dataframe: Input DataFrame
            date_column: Name of the date column

        Raises:
            ValueError: If date_column is not in DataFrame columns
        """
        if date_column not in dataframe.columns:
            raise ValueError(f"Column '{date_column}' not found in DataFrame")
        self._df = dataframe.copy()  # Store a copy to prevent external modifications
        self._date_col = date_column

    def _validate_state(self) -> None:
        """Validate that DataFrame and date column are set."""
        if self._df is None or self._date_col is None:
            raise ValueError("DataFrame or date column not set")

    def get_max_date(self) -> pd.Timestamp:
        """Get the most recent date in the date column.

        Returns:
            pd.Timestamp: Maximum date

        Raises:
            ValueError: If DataFrame or date column not set
        """
        self._validate_state()
        return self._df[self._date_col].max()

    def get_date_minus(self, days: int) -> pd.Timestamp:
        """Get the date 'days' before the maximum date.

        Args:
            days: Number of days to subtract

        Returns:
            pd.Timestamp: Calculated date
        """
        return self.get_max_date() - timedelta(days=days)

    def calculate_avg_by_date(
        self, date: pd.Timestamp, avg_columns: list[str], keep_columns: list[str]
    ) -> pd.Series:
        """Calculate averages for a specific date.

        Args:
            date: Target date
            avg_columns: Columns to average
            keep_columns: Columns to retain first value

        Returns:
            pd.Series: Combined results with kept values and averages
        """
        self._validate_state()

        # Filter once and reuse
        date_df = self._df[self._df[self._date_col] == date]

        if date_df.empty:
            return pd.Series(index=keep_columns + avg_columns, dtype=float)

        # Calculate averages and kept values efficiently
        avg_values = date_df[avg_columns].mean()
        kept_values = (
            date_df[keep_columns].iloc[0]
            if not date_df[keep_columns].empty
            else pd.Series(index=keep_columns)
        )

        return pd.concat([kept_values, avg_values])

    def calculate_all_averages(
        self, avg_columns: list[str], keep_columns: list[str], num_days: int = 3
    ) -> pd.DataFrame:
        """Calculate averages for multiple days plus total average.

        Args:
            avg_columns: Columns to average
            keep_columns: Columns to retain
            num_days: Number of days to look back (default: 3)

        Returns:
            pd.DataFrame: Results with daily and total averages
        """
        self._validate_state()

        # Get unique dates efficiently
        unique_dates = self._df[self._date_col].unique()
        max_date = self.get_max_date()
        target_dates = [max_date - timedelta(days=i) for i in range(num_days)][::-1]
        valid_dates = [d for d in target_dates if d in unique_dates]

        if not valid_dates:
            return pd.DataFrame(
                columns=keep_columns + [f"Avg {col}" for col in avg_columns]
            )

        # Calculate averages for valid dates
        results = [
            self.calculate_avg_by_date(date, avg_columns, keep_columns)
            for date in valid_dates
        ]
        result_df = pd.DataFrame(results)

        # Calculate total average
        total_avg = pd.Series(
            index=keep_columns,
            data=["" if col == self._date_col else "Average" for col in keep_columns],
        )
        total_avg = pd.concat([total_avg, result_df[avg_columns].mean()])

        # Combine results
        final_df = pd.concat([result_df, pd.DataFrame([total_avg])], ignore_index=True)
        final_df.columns = keep_columns + [f"Avg {col}" for col in avg_columns]

        return final_df


class QueryManager:
    def __init__(self, engine):
        self.engine = engine

    def _fetch_data(
        self, query: str, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Execute SQL query and return DataFrame, handling errors gracefully."""
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def _build_like_conditions(
        self, field: str, values: list[str]
    ) -> tuple[str, dict[str, str]]:
        """Build SQL LIKE conditions and parameters for multiple values."""
        conditions = " OR ".join(f'"{field}" LIKE :{i}' for i in range(len(values)))
        params = {f"{i}": f"%{value}%" for i, value in enumerate(values)}
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
        params = {
            "siteid": f"%{siteid}%",
            "start_date": start_date,
            "end_date": end_date,
        }

        query_parts = [
            "SELECT * FROM ltedaily",
            'WHERE "siteid" LIKE :siteid',
            'AND "date" BETWEEN :start_date AND :end_date',
        ]

        if band:
            if len(band) == 1:
                query_parts.append('AND "band" LIKE :band')
                params["band"] = f"%{band[0]}%"
            else:
                placeholders = ", ".join(f":band_{i}" for i in range(len(band)))
                query_parts.append(f'AND "band" IN ({placeholders})')
                params.update({f"band_{i}": b for i, b in enumerate(band)})

        return _self._fetch_data(text(" ".join(query_parts)), params)

    @st.cache_data(ttl=600)
    def get_nbr_data(
        _self,
        siteid: list[str] | str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch LTE daily data with optional multiple siteid filter."""
        params = {
            "start_date": start_date,
            "end_date": end_date,
        }

        query_parts = [
            "SELECT * FROM ltedaily",
            'WHERE "date" BETWEEN :start_date AND :end_date',
        ]

        if isinstance(siteid, str):
            query_parts.append('AND "siteid" LIKE :siteid')
            params["siteid"] = f"%{siteid}%"
        else:
            placeholders = ", ".join(f":siteid_{i}" for i in range(len(siteid)))
            query_parts.append(f'AND "siteid" IN ({placeholders})')
            params.update({f"siteid_{i}": s for i, s in enumerate(siteid)})

        return _self._fetch_data(text(" ".join(query_parts)), params)

    @st.cache_data(ttl=600)
    def get_ltedaily_wo_band(
        _self, siteid: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch LTE daily data without band filter."""
        query = text("""
            SELECT * FROM ltedaily
            WHERE "siteid" LIKE :siteid
            AND "date" BETWEEN :start_date AND :end_date
        """)
        params = {
            "siteid": f"%{siteid}%",
            "start_date": start_date,
            "end_date": end_date,
        }
        return _self._fetch_data(query, params)

    @st.cache_data(ttl=600)
    def get_gsmdaily(
        _self, siteid: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch GSM daily data"""
        query = text("""
            SELECT
            "date",
            "siteid",
            "cellname",
            "TCH Traffic",
            "SDCCH Traffic",
            "GPRS Payload (Mbyte)",
            "EDGE Payload (Mbyte)"
            FROM gsmdaily
            WHERE "siteid" LIKE :siteid
            AND "date" BETWEEN :start_date AND :end_date
        """)
        params = {
            "siteid": f"%{siteid}%",
            "start_date": start_date,
            "end_date": end_date,
        }
        return _self._fetch_data(query, params)

    @st.cache_data(ttl=600)
    def get_gsmdaily_cluster(
        _self,
        siteid: list[str] | str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch GSM daily data cluster.

        Args:
            _self: Instance of the class (assuming this is a method)
            siteid: Single site ID string or list of site IDs
            start_date: Start date in string format
            end_date: End date in string format

        Returns:
            pd.DataFrame: DataFrame containing GSM daily data
        """
        params = {
            "start_date": start_date,
            "end_date": end_date,
        }

        query_parts = [
            "SELECT * FROM gsmdaily",
            'WHERE "date" BETWEEN :start_date AND :end_date',
        ]

        if isinstance(siteid, str):
            query_parts.append('AND "siteid" LIKE :siteid')
            params["siteid"] = f"%{siteid}%"
        else:
            placeholders = ", ".join(f":siteid_{i}" for i in range(len(siteid)))
            query_parts.append(f'AND "siteid" IN ({placeholders})')
            params.update({f"siteid_{i}": s for i, s in enumerate(siteid)})

        final_query = " ".join(query_parts)
        return _self._fetch_data(text(final_query), params)

    @st.cache_data(ttl=600)
    def get_ltehourly_data(_self, siteids: list[str]) -> pd.DataFrame:
        """Fetch LTE hourly data for multiple site IDs."""
        conditions, params = _self._build_like_conditions("eNodeB Name", siteids)
        query = text(f"""
            SELECT * FROM ltehourly
            WHERE ({conditions})
        """)
        df = _self._fetch_data(query, params)
        return (
            df.assign(
                Time=lambda x: pd.to_datetime(x["Time"], format="%Y-%m-%d %H:%M:%S.%f")
            )
            if not df.empty
            else df
        )

    @st.cache_data(ttl=600)
    def get_ltetastate_data(_self, siteids: list[str]) -> pd.DataFrame:
        """Fetch LTE TA state data for multiple site IDs with the most recent date."""
        conditions, params = _self._build_like_conditions("enodeb_name", siteids)
        query = text(f"""
            SELECT
                "enodeb_name",
                "cell_name",
                "localcell_id",
                "l_ta_ue_index0",
                "l_ta_ue_index1",
                "l_ta_ue_index2",
                "l_ta_ue_index3",
                "l_ta_ue_index4",
                "l_ta_ue_index5",
                "l_ta_ue_index6",
                "l_ta_ue_index7",
                "l_ta_ue_index8",
                "l_ta_ue_index9",
                "l_ta_ue_index10",
                "l_ta_ue_index11",
                "l_ta_ue_index12",
                "l_ta_ue_index13",
                "l_ta_ue_index14",
                "l_ta_ue_index15"
            FROM timingadvance
            WHERE ({conditions})
            AND date = (SELECT MAX(date) FROM timingadvance WHERE ({conditions}))
        """)
        return _self._fetch_data(query, params)


class ChartGenerator:
    def __init__(self):
        self.color_palette = ColorPalette()

    def get_colors(self, num_colors):
        return [self.color_palette.get_color(i) for i in range(num_colors)]

    def get_header(self, cells: list[str]) -> str:
        sectors = sorted({f"Sector {cell[-1]}" for cell in cells if cell})
        return ", ".join(sectors) or "No Sectors"

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

            unique_last_digits = sorted(
                {
                    cell[-1]
                    for cell in sector_data[cell_name].unique()
                    if cell and cell[-1].isdigit()
                },
                key=int,
            )
            title = (
                ", ".join(f"SECTOR {digit}" for digit in unique_last_digits)
                if unique_last_digits
                else "No Sectors"
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
                            if isinstance(yline, str) and yline in df.columns:
                                yline_value = sector_data[yline].mean()
                            else:
                                try:
                                    yline_value = float(yline)
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

                        if xrule:
                            fig.add_vline(
                                x=st.session_state["xrule"],
                                line_width=2,
                                line_dash="dash",
                                line_color="#808080",
                            )

                        fig.update_layout(
                            margin=dict(t=20, l=20, r=20, b=20),
                            title_text=title,
                            title_x=0.4,
                            template="plotly_white",
                            hoverlabel=dict(
                                font_size=14, font_family="Plus Jakarta Sans Light"
                            ),
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
                            yaxis=dict(tickfont=dict(size=14, color="#000000")),
                            xaxis=dict(tickfont=dict(size=14, color="#000000")),
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

        custom_headers = {
            "rrc_setup_sr_service": "RRC Setup Success Rate",
            "erab_setup_sr_all": "ERAB Setup Success Rate",
            "call_setup_sr": "Call Setup Success Rate",
            "service_drop_rate": "Service Drop Rate",
            "intrafreq_ho_out_sr": "Intra-Frequency Handover Success Rate",
            "inter_frequency_handover_sr": "Inter-Frequency Handover Success Rate",
            "lte_to_geran_redirection_sr": "LTE to GERAN Redirection Success Rate",
            "csfb_execution_sr": "CSFB Execution Success Rate",
            "csfb_preparation_sr": "CSFB Preparation Success Rate",
            "uplink_interference": "Uplink Interference",
            "radio_network_availability_rate": "Radio Network Availability",
            "cqi": "Channel Quality Indicator (CQI)",
            "se": "Spectral Efficiency (SE)",
            "total_traffic_volume_gb": "Total Traffic Volume (GB)",
            "total_user": "Total Users",
            "cell_dl_avg_throughput_mbps": "Cell DL Avg Throughput (Mbps)",
            "cell_ul_avg_throughput_mbps": "Cell UL Avg Throughput (Mbps)",
            "user_dl_avg_throughput_mbps": "User DL Avg Throughput (Mbps)",
            "user_ul_avg_throughput_mbps": "User UL Avg Throughput (Mbps)",
            "active_user": "Active Users",
            "ta_average": "Timing Advance Average",
        }

        df["date"] = pd.to_datetime(df["date"])

        if "siteid" in df.columns and "sector" in df.columns:
            df["cellsector"] = (
                df["neid"].astype(str) + "_S" + df["sector"].astype(int).astype(str)
            )
        else:
            st.warning(
                "Kolom 'siteid' atau 'sector' tidak ditemukan. Menggunakan 'cell_name' sebagai fallback."
            )
            df["cellsector"] = df["cell_name"]

        unique_cells = df["cellsector"].unique()
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
                            y=alt.Y(
                                f"{param}:Q",
                                title="",
                                scale=alt.Scale(zero=False),
                            ),
                            color=alt.Color(
                                "cellsector:N",
                                scale=alt.Scale(
                                    domain=list(color_mapping.keys()),
                                    range=list(color_mapping.values()),
                                ),
                                legend=alt.Legend(
                                    title=None,
                                    orient="bottom",
                                    columns=5,
                                    direction="horizontal",
                                    columnPadding=-4,
                                    symbolLimit=0,
                                    labelLimit=0,
                                    labelFontSize=12,
                                    labelColor="black",
                                ),
                            ),
                            tooltip=[
                                "cellsector",
                                "date",
                                param,
                            ],
                        )
                        .properties(
                            title=custom_headers.get(
                                param, param.replace("_", " ").title().upper()
                            ),
                            height=400,
                            width="container",
                        )
                        .interactive()
                    )

                    st.altair_chart(chart, use_container_width=True)

    def create_charts_for_stacked_area(self, df, cell_name, x_param, y_param):
        df = df.sort_values(by=x_param)
        df[y_param] = df[y_param].astype(float)
        df["sector"] = df[cell_name].apply(self.determine_sector)
        color_mapping = {
            cell: color
            for cell, color in zip(
                df[cell_name].unique(),
                self.get_colors(len(df[cell_name].unique())),
            )
        }

        sector_count = df["sector"].nunique()
        cols = min(sector_count, 3)
        columns = st.columns(cols)

        for idx, sector in enumerate(sorted(df["sector"].unique())):
            sector_data = df[df["sector"] == sector]
            unique_last_digits = sorted(
                {
                    cell[-1]
                    for cell in sector_data[cell_name].unique()
                    if cell and cell[-1].isdigit()
                },
                key=int,
            )
            title = (
                ", ".join(f"SECTOR {digit}" for digit in unique_last_digits)
                if unique_last_digits
                else "No Sectors"
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
                        fig = px.area(
                            sector_data,
                            x=x_param,
                            y=y_param,
                            color=cell_name,
                            color_discrete_map=color_mapping,
                            hover_data={
                                cell_name: True,
                                y_param: True,
                            },
                        )
                        fig.update_traces(
                            hovertemplate=f"<b>{cell_name}:</b> %{{customdata[0]}}<br><b>{y_param}:</b> %{{y}}<extra></extra>"
                        )

                        fig.update_layout(
                            margin=dict(t=20, l=20, r=20, b=20),
                            title_text=title,
                            title_x=0.4,
                            xaxis_title=None,
                            yaxis_title=None,
                            template="plotly_white",
                            hoverlabel=dict(
                                font_size=16, font_family="Plus Jakarta Sans Light"
                            ),
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.5,
                                xanchor="center",
                                x=0.5,
                                itemclick="toggleothers",
                                itemdoubleclick="toggle",
                                itemsizing="constant",
                                font=dict(size=16),
                                title=None,
                            ),
                            yaxis=dict(
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
                            paper_bgcolor="#F5F5F5",
                            plot_bgcolor="#F5F5F5",
                            width=600,
                            height=250,
                            showlegend=True,
                        )

                        st.plotly_chart(fig, use_container_width=True)

    def create_charts_for_stacked_area_neid(self, df, neid, x_param, y_param, key=None):
        df[y_param] = df[y_param].astype(float)
        df_agg = df.groupby([x_param, neid], as_index=False)[y_param].sum()

        df_agg = df_agg.sort_values(by=x_param)

        color_mapping = {
            cell: color
            for cell, color in zip(
                df_agg[neid].unique(),
                self.get_colors(len(df_agg[neid].unique())),
            )
        }

        with stylable_container(
            key=f"container_with_border_{neid}",
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
                fig = px.area(
                    df_agg,
                    x=x_param,
                    y=y_param,
                    color=neid,
                    color_discrete_map=color_mapping,
                    hover_data={neid: True, y_param: True},
                )

                fig.update_traces(hovertemplate=f"<b>{neid}:</b>%{{y}}<extra></extra>")

                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title=None,
                    margin=dict(t=20, l=20, r=20, b=20),
                    hoverlabel=dict(
                        font_size=16, font_family="Plus Jakarta Sans Light"
                    ),
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        itemclick="toggleothers",
                        itemdoubleclick="toggle",
                        itemsizing="constant",
                        font=dict(size=16),
                        title=None,
                    ),
                    yaxis=dict(
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
                    paper_bgcolor="#F5F5F5",
                    plot_bgcolor="#F5F5F5",
                    width=600,
                    height=350,
                    showlegend=True,
                )

                st.plotly_chart(fig, use_container_width=True, key=key)

    def create_charts_timingadvance(self, df, sector):
        all_plot_columns = [
            "l_ta_ue_index0",
            "l_ta_ue_index1",
            "l_ta_ue_index2",
            "l_ta_ue_index3",
            "l_ta_ue_index4",
            "l_ta_ue_index5",
            "l_ta_ue_index6",
            "l_ta_ue_index7",
            "l_ta_ue_index8",
            "l_ta_ue_index9",
            "l_ta_ue_index10",
            "l_ta_ue_index11",
            "l_ta_ue_index12",
            "l_ta_ue_index13",
            "l_ta_ue_index14",
            "l_ta_ue_index15",
        ]

        available_columns = [col for col in all_plot_columns if col in df.columns]
        if not available_columns:
            st.error("No TA index columns found in the data")
            return

        required_columns = ["cell_name"]
        if "localcell_id" in df.columns:
            required_columns.append("localcell_id")
            color_by = "localcell_id"
        else:
            color_by = "cell_name"

        try:
            df = df[required_columns + available_columns]
        except KeyError as e:
            st.error(f"Error selecting columns: {e}")
            return

        df["sector"] = df["cell_name"].apply(self.determine_sector)

        unique_values = df[color_by].unique()
        colors = self.get_colors(len(unique_values))
        color_mapping = {value: color for value, color in zip(unique_values, colors)}

        for sector_num in sorted(df["sector"].unique()):
            sector_data = df[df["sector"] == sector_num]
            unique_last_digits = sorted(
                {
                    cell[-1]
                    for cell in sector_data["cell_name"].unique()
                    if cell and cell[-1].isdigit()
                },
                key=int,
            )
            title = (
                ", ".join(f"SECTOR {digit}" for digit in unique_last_digits)
                if unique_last_digits
                else "No Sectors"
            )

            with stylable_container(
                key=f"container_with_border_ta_{sector_num}",
                css_styles="""
                    {
                        background-color: #F5F5F5;
                        border: 2px solid rgba(49, 51, 63, 0.2);
                        border-radius: 0.5rem;
                        padding: calc(1em - 1px);
                        margin-bottom: 1rem;
                    }
                    """,
            ):
                container = st.container()
                with container:
                    fig = go.Figure()

                    for value in sector_data[color_by].unique():
                        filtered_df = sector_data[sector_data[color_by] == value]
                        cell_name = filtered_df["cell_name"].iloc[0]
                        fig.add_trace(
                            go.Bar(
                                x=available_columns,
                                y=filtered_df.loc[:, available_columns].values[0],
                                name=cell_name,
                                marker_color=color_mapping[value],
                                hovertemplate=(
                                    f"<b>®️ {cell_name}</b><br>"
                                    f"<b></b> %{{x}} - %{{y}}<br>"
                                    "<extra></extra>"
                                ),
                                hoverlabel=dict(
                                    font_size=16, font_family="Plus Jakarta Sans Light"
                                ),
                            )
                        )

                    fig.update_layout(
                        barmode="group",
                        xaxis_title=None,
                        yaxis_title=None,
                        plot_bgcolor="#F5F5F5",
                        paper_bgcolor="#F5F5F5",
                        height=400,
                        width=800,
                        title_text=title,
                        title_x=0.4,
                        font=dict(
                            family="Plus Jakarta Sans Light", size=25, color="#717577"
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            bgcolor="#F5F5F5",
                            bordercolor="#F5F5F5",
                            itemclick="toggleothers",
                            itemdoubleclick="toggle",
                            itemsizing="constant",
                            font=dict(size=14),
                        ),
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis=dict(
                            tickfont=dict(size=12, color="#000000"),
                        ),
                        xaxis=dict(
                            tickfont=dict(size=12, color="#000000"),
                        ),
                    )

                    st.plotly_chart(fig, use_container_width=True)

    def create_charts_stacked(self, df, neid, x_param, y_param, key=None):
        df[y_param] = df[y_param].astype(float)
        df_agg = df.groupby([x_param, neid], as_index=False)[y_param].sum()
        df_agg = df_agg.sort_values(by=x_param)

        color_mapping = {
            cell: color
            for cell, color in zip(
                df_agg[neid].unique(),
                self.get_colors(len(df_agg[neid].unique())),
            )
        }

        container = st.container(border=True)
        with container:
            chart = (
                alt.Chart(df_agg)
                .mark_area()
                .encode(
                    x=alt.X(
                        f"{x_param}:T",
                        title=None,
                        axis=alt.Axis(
                            format="%Y-%m-%d",  # Show year, month, and day
                            labelAngle=45,  # Rotate labels for better readability
                            labelFontSize=14,
                            labelColor="#000000",
                        ),
                    ),
                    y=alt.Y(
                        f"{y_param}:Q",
                        title=None,
                        stack=True,
                    ),
                    color=alt.Color(
                        f"{neid}:N",
                        scale=alt.Scale(
                            domain=list(color_mapping.keys()),
                            range=list(color_mapping.values()),
                        ),
                        legend=alt.Legend(
                            title=None,
                            orient="bottom",
                            columns=4,
                            direction="horizontal",
                            columnPadding=10,
                            symbolLimit=0,
                            labelLimit=0,
                            labelFontSize=14,
                            labelColor="black",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(f"{neid}:N", title=neid),
                        alt.Tooltip(f"{y_param}:Q", title=y_param),
                        alt.Tooltip(f"{x_param}:T", title=x_param, format="%Y-%m-%d"),
                    ],
                )
                .properties(
                    width=600,
                    height=350,
                )
                .configure_view(
                    strokeWidth=0,
                    fill="#F5F5F5",
                )
                .configure_axis(
                    labelFontSize=14,
                    labelColor="#000000",
                    grid=False,
                )
                .configure_legend(
                    labelFont="Plus Jakarta Sans Light",
                    labelFontSize=16,
                    padding=10,
                )
                .interactive()
            )

            st.altair_chart(chart, use_container_width=True, key=key)

    def create_charts_line(self, df, neid, x_param, y_param, key=None):
        df[y_param] = df[y_param].astype(float)
        df_agg = df.groupby([x_param, neid], as_index=False)[y_param].sum()
        df_agg = df_agg.sort_values(by=x_param)

        color_mapping = {
            cell: color
            for cell, color in zip(
                df_agg[neid].unique(),
                self.get_colors(len(df_agg[neid].unique())),
            )
        }

        container = st.container(border=True)
        with container:
            chart = (
                alt.Chart(df_agg)
                .mark_line()  # Changed from mark_area to mark_line for line chart
                .encode(
                    x=alt.X(
                        f"{x_param}:T",
                        title=None,
                        axis=alt.Axis(
                            format="%Y-%m-%d",  # Show year, month, and day
                            labelAngle=45,  # Rotate labels for better readability
                            labelFontSize=14,
                            labelColor="#000000",
                        ),
                    ),
                    y=alt.Y(
                        f"{y_param}:Q",
                        title=None,
                        stack=None,  # Remove stacking for line chart
                    ),
                    color=alt.Color(
                        f"{neid}:N",
                        scale=alt.Scale(
                            domain=list(color_mapping.keys()),
                            range=list(color_mapping.values()),
                        ),
                        legend=alt.Legend(
                            title=None,
                            orient="bottom",
                            columns=4,
                            direction="horizontal",
                            columnPadding=10,
                            symbolLimit=0,
                            labelLimit=0,
                            labelFontSize=14,
                            labelColor="black",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(f"{neid}:N", title=neid),
                        alt.Tooltip(f"{y_param}:Q", title=y_param),
                        alt.Tooltip(f"{x_param}:T", title=x_param, format="%Y-%m-%d"),
                    ],
                )
                .properties(
                    width=600,
                    height=350,
                )
                .configure_view(
                    strokeWidth=0,
                    fill="#F5F5F5",
                )
                .configure_axis(
                    labelFontSize=14,
                    labelColor="#000000",
                    grid=False,
                )
                .configure_legend(
                    labelFont="Plus Jakarta Sans Light",
                    labelFontSize=16,
                    padding=10,
                )
                .interactive()
            )

            st.altair_chart(chart, use_container_width=True, key=key)


class App:
    def __init__(self):
        self.config = Config().load()
        self.database_session = DatabaseSession(self.config)
        self.query_manager = None
        self.avg_calc = DateCalc()
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

            selected_nbr = self.streamlit_interface.nbr_selection(sitelist)
            st.session_state.selected_nbr = selected_nbr

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

                            if selected_nbr:
                                df_nbr = self.query_manager.get_nbr_data(
                                    selected_nbr, start_date_str, end_date_str
                                )
                                df_daily_all_copy = df_daily_all.copy()
                                df_nbr_copy = df_nbr.copy()
                                enodeb_name = (
                                    df_daily_all["enodeb_name"].iloc[0]
                                    if not df_daily_all.empty
                                    and "enodeb_name" in df_daily_all.columns
                                    else site
                                )
                                df_daily_all_copy["sitesow"] = enodeb_name
                                df_nbr_copy["sitesow"] = enodeb_name
                                df_sitesow = pd.concat(
                                    [df_daily_all_copy, df_nbr_copy], ignore_index=True
                                )
                                self.dataframe_manager.add_dataframe(
                                    f"sitesow_{site}", df_sitesow
                                )
                            else:
                                df_daily_all_copy = df_daily_all.copy()
                                enodeb_name = (
                                    df_daily_all["enodeb_name"].iloc[0]
                                    if not df_daily_all.empty
                                    and "enodeb_name" in df_daily_all.columns
                                    else site
                                )
                                df_daily_all_copy["sitesow"] = enodeb_name
                                self.dataframe_manager.add_dataframe(
                                    f"sitesow_{site}", df_daily_all_copy
                                )

                            df_nbr = self.query_manager.get_nbr_data(
                                selected_nbr if selected_nbr else [site],
                                start_date_str,
                                end_date_str,
                            )
                            self.dataframe_manager.add_dataframe(f"nbr_{site}", df_nbr)

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
                            df_gsm = self.query_manager.get_gsmdaily(
                                site, start_date_str, end_date_str
                            )
                            self.dataframe_manager.add_dataframe(
                                f"gsmdaily_{site}", df_gsm
                            )
                            df_gsm_cluster = self.query_manager.get_gsmdaily_cluster(
                                site, start_date_str, end_date_str
                            )
                            self.dataframe_manager.add_dataframe(
                                f"gsmdaily_cluster_{site}", df_gsm_cluster
                            )

        if any(
            key.startswith("dailysow_") for key in self.dataframe_manager.dataframes
        ):
            tab1, tab2, tab3, tab4 = st.tabs(
                ["SOW SITE", "COLO & ClUSTER", "Payload COLLO & ClUSTER", "TA"]
            )

            with tab1:
                for site in selected_sites:
                    df_daily_sow = self.dataframe_manager.get_dataframe(
                        f"dailysow_{site}"
                    )
                    df_daily_all = self.dataframe_manager.get_dataframe(
                        f"dailyall_{site}"
                    )
                    df_daily_nbr = self.dataframe_manager.get_dataframe(f"nbr_{site}")
                    df_sitesow = self.dataframe_manager.get_dataframe(f"sitesow_{site}")
                    st.markdown(
                        *styling(
                            f"OSS Performance - {site}",
                            font_size=28,
                            text_align="Center",
                            background_color="#DC0013",
                            tag="h6",
                            font_color="#FFFFFF",
                        )
                    )
                    sac.divider(
                        align="center",
                        size="s",
                    )
                    st.markdown(
                        *styling(
                            "📶 KPI Performance 3 Days (New)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    if df_daily_all is not None and not df_daily_all.empty:
                        self.avg_calc.set_dataframe(df_daily_all, "date")
                        avg_result = self.avg_calc.calculate_all_averages(
                            [
                                "rrc_setup_sr_service",
                                "erab_setup_sr_all",
                                "call_setup_sr",
                                "csfb_execution_sr",
                                "csfb_preparation_sr",
                                "service_drop_rate",
                                "intrafreq_ho_out_sr",
                                "inter_frequency_handover_sr",
                                "lte_to_geran_redirection_sr",
                                "uplink_interference",
                                "user_dl_avg_throughput_mbps",
                                "cqi",
                                "se",
                            ],
                            ["enodeb_name", "region", "date"],
                            num_days=3,
                        )
                        html_table = "<table class='custom-table'>"
                        html_table += (
                            "<thead><tr>"
                            + "".join(f"<th>{col}</th>" for col in avg_result.columns)
                            + "</tr></thead>"
                        )
                        html_table += "<tbody>"
                        for row in avg_result.itertuples(index=False):
                            html_table += (
                                "<tr>"
                                + "".join(f"<td>{val}</td>" for val in row)
                                + "</tr>"
                            )
                        html_table += "</tbody></table>"

                        st.markdown(
                            """
                            <style>
                            .custom-table {
                                font-size: 10px !important;
                                font-family: Arial, sans-serif !important;
                                border-collapse: collapse !important;
                                text-align: center !important;
                                width: 100% !important;
                            }
                            .custom-table th {
                                background-color: #F5F5F5 !important;
                                border: 1px solid #ddd !important;
                                padding: 2px !important;
                                text-align: center !important;
                                vertical-align: top !important;
                                height: 20px !important;
                            }
                            .custom-table td {
                                border: 1px solid #ddd !important;
                                padding: 2px !important;
                                vertical-align: top !important;
                                text-align: center !important;
                                height: 20px !important;
                            }
                            .custom-table tr {
                                height: 20px !important;
                            }
                            .custom-table tr:nth-child(even) {
                                background-color: #f9f9f9 !important;
                            }
                            .custom-table tr:hover {
                                background-color: #f5f5f5 !important;
                            }
                            </style>
                            """
                            + html_table,
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        *styling(
                            "📶 KPI Performance 5 Days (1st tier)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    if df_sitesow is not None and not df_sitesow.empty:
                        self.avg_calc.set_dataframe(df_sitesow, "date")
                        avg_result = self.avg_calc.calculate_all_averages(
                            [
                                "rrc_setup_sr_service",
                                "erab_setup_sr_all",
                                "call_setup_sr",
                                "csfb_execution_sr",
                                "csfb_preparation_sr",
                                "service_drop_rate",
                                "intrafreq_ho_out_sr",
                                "inter_frequency_handover_sr",
                                "lte_to_geran_redirection_sr",
                                "uplink_interference",
                                "user_dl_avg_throughput_mbps",
                                "cqi",
                                "se",
                            ],
                            ["sitesow", "region", "date"],
                            num_days=5,
                        )
                        html_table = "<table class='custom-table'>"
                        html_table += (
                            "<thead><tr>"
                            + "".join(f"<th>{col}</th>" for col in avg_result.columns)
                            + "</tr></thead>"
                        )
                        html_table += "<tbody>"
                        for row in avg_result.itertuples(index=False):
                            html_table += (
                                "<tr>"
                                + "".join(f"<td>{val}</td>" for val in row)
                                + "</tr>"
                            )
                        html_table += "</tbody></table>"

                        st.markdown(
                            """
                            <style>
                            .custom-table {
                                font-size: 10px !important;
                                font-family: Arial, sans-serif !important;
                                border-collapse: collapse !important;
                                text-align: center !important;
                                width: 100% !important;
                            }
                            .custom-table th {
                                background-color: #F5F5F5 !important;
                                border: 1px solid #ddd !important;
                                padding: 2px !important;
                                text-align: center !important;
                                vertical-align: top !important;
                                height: 20px !important;
                            }
                            .custom-table td {
                                border: 1px solid #ddd !important;
                                padding: 2px !important;
                                vertical-align: top !important;
                                text-align: center !important;
                                height: 20px !important;
                            }
                            .custom-table tr {
                                height: 20px !important;
                            }
                            .custom-table tr:nth-child(even) {
                                background-color: #f9f9f9 !important;
                            }
                            .custom-table tr:hover {
                                background-color: #f5f5f5 !important;
                            }
                            </style>
                            """
                            + html_table,
                            unsafe_allow_html=True,
                        )

                    st.markdown(*styling(""))
                    st.markdown(
                        *styling(
                            f"KPI Trend Chart (1 Week) - {site}",
                            font_size=28,
                            text_align="Center",
                            background_color="#DC0013",
                            tag="h6",
                            font_color="#FFFFFF",
                        )
                    )
                    st.markdown(*styling(""))
                    df_hourly = self.dataframe_manager.get_dataframe(f"hourly_{site}")
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
                        st.markdown(
                            *styling(
                                f"📶 Intra HO {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "intrafreq_ho_out_sr",
                            xrule=False,
                            yline="98.0",
                        )
                        st.markdown(
                            *styling(
                                f"📶 Inter HO {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "inter_frequency_handover_sr",
                            xrule=False,
                            yline="98.0",
                        )
                        sac.divider(
                            label="PAGE 2",
                            align="center",
                            size="xl",
                            color="#DC0013",
                        )
                        st.markdown(
                            *styling(
                                f"📶 L2G SR {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "lte_to_geran_redirection_sr",
                            xrule=False,
                            yline="99.9",
                        )
                        st.markdown(
                            *styling(
                                f"📶 RSSI {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "uplink_interference",
                            xrule=False,
                            yline="-100",
                        )
                        st.markdown(
                            *styling(
                                f"📶 Availability {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "radio_network_availability_rate",
                            xrule=False,
                        )
                        st.markdown(
                            *styling(
                                f"📶 CQI {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "cqi",
                            xrule=False,
                            yline="9",
                        )
                        st.markdown(
                            *styling(
                                f"📶 SE {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "se",
                            xrule=False,
                            yline="1.4",
                        )
                        sac.divider(
                            label="PAGE 3",
                            align="center",
                            size="xl",
                            color="#DC0013",
                        )
                        st.markdown(
                            *styling(
                                f"📶 User Throughput {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "user_dl_avg_throughput_mbps",
                            xrule=False,
                        )
                        st.markdown(
                            *styling(
                                f"📶 Cell Throughput {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "cell_dl_avg_throughput_mbps",
                            xrule=False,
                        )
                        st.markdown(
                            *styling(
                                f"📶 Active User {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_daily_sow,
                            "cell_name",
                            "date",
                            "active_user",
                            xrule=False,
                        )
                        st.markdown(
                            *styling(
                                f"📶 PRB Utilization {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_daily(
                            df_hourly,
                            "Cell Name",
                            "Time",
                            "DL Resource Block Utilizing Rate %_FIX",
                            xrule=False,
                        )
                        st.markdown(*styling(""))
                        st.markdown(
                            *styling(
                                f"Payload - {site}",
                                font_size=28,
                                text_align="Center",
                                background_color="#DC0013",
                                tag="h6",
                                font_color="#FFFFFF",
                            )
                        )
                        st.markdown(*styling(""))
                        st.markdown(
                            *styling(
                                f"📶 Payload SOW {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_stacked_area(
                            df=df_daily_sow,
                            cell_name="cell_name",
                            x_param="date",
                            y_param="total_traffic_volume_gb",
                        )
                        st.markdown(
                            *styling(
                                f"📶 Payload Sector All System  {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_stacked_area(
                            df=df_daily_all,
                            cell_name="cell_name",
                            x_param="date",
                            y_param="total_traffic_volume_gb",
                        )
                        st.markdown(
                            *styling(
                                f"📶 Total Payload {site}",
                                font_size=24,
                                text_align="left",
                                tag="h6",
                            )
                        )
                        self.chart_generator.create_charts_for_stacked_area_neid(
                            df=df_daily_all,
                            neid="neid",
                            x_param="date",
                            y_param="total_traffic_volume_gb",
                            key=f"stacked_area_neid_tab1_{site}",  # Unique key added
                        )

            with tab2:
                for site in selected_sites:
                    df_daily_all = self.dataframe_manager.get_dataframe(
                        f"dailyall_{site}"
                    )
                    df_sitesow = self.dataframe_manager.get_dataframe(f"sitesow_{site}")

                    df_to_use = df_sitesow if df_sitesow is not None else df_daily_all

                    if df_to_use is not None:
                        self.chart_generator.create_daily_charts(df_to_use, site)

            with tab3:
                for site in selected_sites:
                    lte_cosite = self.dataframe_manager.get_dataframe(
                        f"dailyall_{site}"
                    )
                    gsm_cosite = self.dataframe_manager.get_dataframe(
                        f"gsmdaily_{site}"
                    )

                    lte_cluster = (
                        self.dataframe_manager.get_dataframe(f"nbr_{site}")
                        if selected_nbr
                        else self.dataframe_manager.get_dataframe(f"dailyall_{site}")
                    )
                    gsm_cluster = (
                        self.dataframe_manager.get_dataframe(f"gsmdaily_cluster_{site}")
                        if selected_nbr
                        else self.dataframe_manager.get_dataframe(f"gsmdaily_{site}")
                    )
                    st.write(gsm_cluster)
                    st.write(lte_cluster)
                    if (
                        selected_nbr
                        and "Run Query" in st.session_state
                        and st.session_state["Run Query"]
                    ):
                        start_date, end_date = date_range
                        start_date_str = start_date.strftime("%Y-%m-%d")
                        end_date_str = end_date.strftime("%Y-%m-%d")
                        df_nbr = self.query_manager.get_nbr_data(
                            selected_nbr,
                            start_date_str,
                            end_date_str,
                        )
                        df_gsm_cluster = self.query_manager.get_gsmdaily_cluster(
                            selected_nbr,
                            start_date_str,
                            end_date_str,
                        )
                        self.dataframe_manager.add_dataframe(f"nbr_{site}", df_nbr)
                        self.dataframe_manager.add_dataframe(
                            f"gsmdaily_cluster_{site}", df_gsm_cluster
                        )

                    # if lte_cluster is None:
                    #     cols = st.columns(1)
                    #     col1 = cols[0]
                    #     col2 = None
                    # else:
                    #     cols = st.columns(2)
                    #     col1, col2 = cols
                    col1 = st.columns(1)
                    # with col1:
                    st.markdown(
                        *styling(
                            f"📶 Payload (Mb) LTE Co-Site {site}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    self.chart_generator.create_charts_stacked(
                        df=lte_cosite,
                        neid="neid",
                        x_param="date",
                        y_param="total_traffic_volume_gb",
                        key=f"stacked_area_neid_cosite_{site}",
                    )
                    st.markdown(
                        *styling(
                            f"📶 TCH Traffic 2G Co-Site {site}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    self.chart_generator.create_charts_line(
                        df=df_gsm,
                        neid="cellname",
                        x_param="date",
                        y_param="TCH Traffic",
                        key=f"stacked_area_neid_cosite_{site}",
                    )
                    st.markdown(
                        *styling(
                            f"📶 SDCCH Traffic 2G Co-Site {site}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    self.chart_generator.create_charts_line(
                        df=df_gsm,
                        neid="cellname",
                        x_param="date",
                        y_param="SDCCH Traffic",
                        key=f"stacked_area_neid_cosite_{site}",
                    )
                    st.markdown(
                        *styling(
                            f"📶 GPRS Payload (Mbyte) 2G Co-Site {site}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    self.chart_generator.create_charts_stacked(
                        df=df_gsm,
                        neid="cellname",
                        x_param="date",
                        y_param="GPRS Payload (Mbyte)",
                        key=f"stacked_area_neid_cosite_{site}",
                    )
                    st.markdown(
                        *styling(
                            f"📶 EDGE Payload (Mbyte) 2G Co-Site {site}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    self.chart_generator.create_charts_stacked(
                        df=df_gsm,
                        neid="cellname",
                        x_param="date",
                        y_param="EDGE Payload (Mbyte)",
                        key=f"stacked_area_neid_cosite_{site}",
                    )
                    # if col2:
                    #     with col2:
                    #         st.markdown(
                    #             *styling(
                    #                 f"📶 Payload (Mb) LTE Cluster {site}",
                    #                 font_size=24,
                    #                 text_align="left",
                    #                 tag="h6",
                    #             )
                    #         )
                    #         self.chart_generator.create_charts_stacked(
                    #             df=lte_cluster,
                    #             neid="siteid",
                    #             x_param="date",
                    #             y_param="total_traffic_volume_gb",
                    #             key=f"stacked_area_neid_cluster_{site}",
                    #         )
                    #         st.markdown(
                    #             *styling(
                    #                 f"📶 TCH Traffic 2G Cluster {site}",
                    #                 font_size=24,
                    #                 text_align="left",
                    #                 tag="h6",
                    #             )
                    #         )
                    #         self.chart_generator.create_charts_line(
                    #             df=gsm_cluster,
                    #             neid="siteid",
                    #             x_param="date",
                    #             y_param="TCH Traffic",
                    #             key=f"gsm_cluster_tch_{site}",
                    #         )
                    #         st.markdown(
                    #             *styling(
                    #                 f"📶 SDCCH Traffic 2G Cluster {site}",
                    #                 font_size=24,
                    #                 text_align="left",
                    #                 tag="h6",
                    #             )
                    #         )
                    #         self.chart_generator.create_charts_line(
                    #             df=gsm_cluster,
                    #             neid="siteid",
                    #             x_param="date",
                    #             y_param="SDCCH Traffic",
                    #             key=f"gsm_cluster_sdcch_{site}",
                    #         )
                    #         st.markdown(
                    #             *styling(
                    #                 f"📶 GPRS Payload (Mbyte) 2G Cluster {site}",
                    #                 font_size=24,
                    #                 text_align="left",
                    #                 tag="h6",
                    #             )
                    #         )
                    #         self.chart_generator.create_charts_stacked(
                    #             df=gsm_cluster,
                    #             neid="siteid",
                    #             x_param="date",
                    #             y_param="GPRS Payload (Mbyte)",
                    #             key=f"gsm_cluster_gprs_{site}",
                    #         )
                    #         st.markdown(
                    #             *styling(
                    #                 f"📶 EDGE Payload (Mbyte) 2G Cluster {site}",
                    #                 font_size=24,
                    #                 text_align="left",
                    #                 tag="h6",
                    #             )
                    #         )
                    #         self.chart_generator.create_charts_stacked(
                    #             df=gsm_cluster,
                    #             neid="siteid",
                    #             x_param="date",
                    #             y_param="EDGE Payload (Mbyte)",
                    #             key=f"gsm_cluster_edge_{site}",
                    #         )

            with tab4:
                for site in selected_sites:
                    df_timingadvance = self.dataframe_manager.get_dataframe(
                        f"tastate_{site}"
                    )

                    st.markdown(
                        *styling(
                            f"📶 Table Timing Advance {site}",
                            font_size=24,
                            text_align="left",
                            tag="h4",
                        )
                    )

                    if df_timingadvance is not None and not df_timingadvance.empty:
                        chart_generator = ChartGenerator()
                        df_timingadvance["sector"] = df_timingadvance[
                            "cell_name"
                        ].apply(chart_generator.determine_sector)
                        df_timingadvance = df_timingadvance.sort_values(by="sector")

                        html_table = "<table class='custom-table'>"
                        html_table += (
                            "<thead><tr>"
                            + "".join(
                                f"<th>{col}</th>" for col in df_timingadvance.columns
                            )
                            + "</tr></thead>"
                        )
                        html_table += "<tbody>"
                        for row in df_timingadvance.itertuples(index=False):
                            html_table += (
                                "<tr>"
                                + "".join(f"<td>{val}</td>" for val in row)
                                + "</tr>"
                            )
                        html_table += "</tbody></table>"

                        st.markdown(
                            """
                            <style>
                            .custom-table {
                                font-size: 10px !important;
                                font-family: Arial, sans-serif !important;
                                border-collapse: collapse !important;
                                text-align: center !important;
                                width: 100% !important;
                            }
                            .custom-table th {
                                background-color: #F5F5F5 !important;
                                border: 1px solid #ddd !important;
                                padding: 2px !important;
                                text-align: center !important;
                                vertical-align: top !important;
                                height: 20px !important;
                            }
                            .custom-table td {
                                border: 1px solid #ddd !important;
                                padding: 2px !important;
                                vertical-align: top !important;
                                text-align: center !important;
                                height: 20px !important;
                            }
                            .custom-table tr {
                                height: 20px !important;
                            }
                            .custom-table tr:nth-child(even) {
                                background-color: #f9f9f9 !important;
                            }
                            .custom-table tr:hover {
                                background-color: #f5f5f5 !important;
                            }
                            </style>
                            """
                            + html_table,
                            unsafe_allow_html=True,
                        )

                        col1 = st.columns(1)[0]
                        with col1:
                            st.markdown(
                                *styling(
                                    f"📶 Graph Timing Advance {site}",
                                    font_size=24,
                                    text_align="left",
                                    tag="h4",
                                )
                            )
                            self.chart_generator.create_charts_timingadvance(
                                df_timingadvance, "cell_name"
                            )
                    else:
                        st.warning(f"No TA state data available for site {site}")


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
