import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_antd_components as sac
import toml
from colors import ColorPalette
from omegaconf import DictConfig, OmegaConf
from plotly.subplots import make_subplots
from review.geoapp import GeoApp
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from streamlit_extras.mandatory_date_range import date_range_picker
from streamlit_extras.stylable_container import stylable_container
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
        data = []
        with open(filepath) as file:
            for line in file:
                columns = line.strip().split(",")
                data.append(columns)
        return data

    def site_selection(self, sitelist):
        column0_data = [row[0] for row in sitelist]
        return st.multiselect("SITEID", column0_data)

    def neid_selection(self, sitelist):
        column2_data = {row[2] for row in sitelist}
        return st.multiselect("NEID", column2_data)

    def select_date_range(self):
        if "date_range" not in st.session_state:
            st.session_state["date_range"] = (
                pd.Timestamp.today() - pd.DateOffset(months=3),
                pd.Timestamp.today(),
            )

        date_range = date_range_picker(
            "DATE RANGE",
            default_start=st.session_state["date_range"][0],
            default_end=st.session_state["date_range"][1],
        )
        return date_range

    def select_xrule_date(self):
        if "xrule_date" not in st.session_state:
            st.session_state["xrule_date"] = pd.Timestamp.today()

        xrule_date = st.date_input("OA Date", st.session_state["xrule_date"])
        st.session_state["xrule_date"] = xrule_date  # Ensure session state is updated
        return xrule_date


class QueryManager:
    def __init__(self, engine):
        self.engine = engine

    def fetch_data(self, query, params=None):
        try:
            df = pd.read_sql(query, self.engine, params=params)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    # def get_mcom_data(self, siteid):
    #     query = text(
    #         """
    #     SELECT "Site_ID", "NODE_ID", "NE_ID", "Cell_Name", "Longitude", "Latitude", "Dir", "Ant_BW",
    #            "Ant_Size", "cellId", "eNBId", "KABUPATEN", "LTE"
    #     FROM mcom
    #     WHERE "Site_ID" LIKE :siteid
    #     """
    #     )
    #     return self.fetch_data(query, {"siteid": siteid})

    # def get_mcom_neid(self, neid):
    #     query = text(
    #         """
    #     SELECT "Site_ID", "NODE_ID", "NE_ID", "Cell_Name", "Longitude", "Latitude", "Dir", "Ant_BW",
    #            "Ant_Size", "cellId", "eNBId", "KABUPATEN", "LTE"
    #     FROM mcom
    #     WHERE "NE_ID" LIKE :neid
    #     """
    #     )
    #     return self.fetch_data(query, {"neid": neid})

    @st.cache_data(ttl=600)
    def get_ltedaily_data(self, siteid, neids, start_date, end_date):
        query = text(
            """
            SELECT *
            FROM ltedaily
            WHERE "siteid" LIKE :siteid
            AND "date" BETWEEN :start_date AND :end_date
            """
        )
        params = {"siteid": siteid, "start_date": start_date, "end_date": end_date}

        if neids:
            neid_conditions = " OR ".join(
                [f'"neid" LIKE :neid_{i}' for i in range(len(neids))]
            )
            query = text(
                f"""
                SELECT *
                FROM ltedaily
                WHERE "siteid" LIKE :siteid
                AND ({neid_conditions})
                AND "date" BETWEEN :start_date AND :end_date
                """
            )
            params.update({f"neid_{i}": neid for i, neid in enumerate(neids)})
        else:
            query = text(
                """
                SELECT *
                FROM ltedaily
                WHERE "siteid" LIKE :siteid
                AND "date" BETWEEN :start_date AND :end_date
                """
            )

        return self.fetch_data(query, params)

    # def get_ltedaily_payload(self, selected_sites, start_date, end_date):
    #     like_conditions = " OR ".join(
    #         [f'"SITEID" LIKE :site_{i}' for i in range(len(selected_sites))]
    #     )
    #     query = text(
    #         f"""
    #         SELECT
    #         "DATE_ID",
    #         "SITEID",
    #         "NEID",
    #         "EutranCell",
    #         "Payload_Total(Gb)",
    #         "CQI Bh"
    #         FROM ltedaily
    #         WHERE ({like_conditions})
    #         AND "DATE_ID" BETWEEN :start_date AND :end_date
    #         """
    #     )
    #     params = {f"site_{i}": f"%{site}%" for i, site in enumerate(selected_sites)}
    #     params.update({"start_date": start_date, "end_date": end_date})

    #     return self.fetch_data(query, params=params)

    @st.cache_data(ttl=600)
    def get_ltehourly_data(_self, selected_sites, end_date):
        like_conditions = " OR ".join(
            [f'"siteid" LIKE :site_{i}' for i in range(len(selected_sites))]
        )
        start_date = pd.Timestamp("today") - pd.Timedelta(days=15)
        query = text(
            f"""
        SELECT
            "date",
            "time",
            "siteid",
            "enodebname",
            "cellname",
            "DL_Resource_Block_Utilizing_Rate(%)",
            "New Active User",
            "User_DL_Avg_Throughput(Kbps)"
        FROM ltehourly
        WHERE ({like_conditions})
        AND "date" BETWEEN :start_date AND :end_date
        """
        )
        params = {f"site_{i}": f"%{site}%" for i, site in enumerate(selected_sites)}
        params.update({"start_date": start_date, "end_date": end_date})

        return _self.fetch_data(query, params=params)

    # def get_target_data(self, city, band):
    #     # def get_target_data(self, city, band):
    #     query = text(
    #         """
    #     SELECT *
    #     FROM target
    #     WHERE "City" = :city AND "Band" = :band
    #     """
    #     )
    #     return self.fetch_data(
    #         # query, {"city": city, "mc_class": mc_class, "band": band}
    #         query,
    #         {"city": city, "band": band},
    #     )

    # @st.cache_data(ttl=600)
    # def get_ltemdt_data(_self, selected_sites):
    #     like_conditions = " OR ".join(
    #         [f'"site" LIKE :site_{i}' for i in range(len(selected_sites))]
    #     )
    #     query = text(
    #         f"""
    #     SELECT
    #     site,
    #     enodebid,
    #     ci,
    #     sample,
    #     rsrp_mean,
    #     rsrq_mean,
    #     rank,
    #     long_grid,
    #     lat_grid
    #     FROM ltemdt
    #     WHERE ({like_conditions})
    #     """
    #     )
    #     params = {f"site_{i}": f"%{site}%" for i, site in enumerate(selected_sites)}
    #     return _self.fetch_data(query, params=params)

    # @st.cache_data(ttl=600)
    # def get_ltetastate_data(_self, siteid):
    #     like_conditions = " OR ".join(
    #         [f'"site" LIKE :site_{i}' for i in range(len(siteid))]
    #     )

    #     query = text(
    #         f"""
    #         SELECT *
    #         FROM ltetastate
    #         WHERE ({like_conditions})
    #         """
    #     )

    #     params = {f"site_{i}": f"%{site}%" for i, site in enumerate(siteid)}
    #     return _self.fetch_data(query, params=params)

    # def get_mcom_tastate(self, selected_neids):
    #     like_conditions = " OR ".join(
    #         [f'"NE_ID" LIKE :neid_{i}' for i in range(len(selected_neids))]
    #     )
    #     query = text(
    #         f"""
    #     SELECT
    #     "Site_ID",
    #     "NE_ID",
    #     "Cell_Name",
    #     "cellId",
    #     "eNBId"
    #     FROM mcom
    #     WHERE ({like_conditions})
    #     """
    #     )
    #     params = {f"neid_{i}": f"%{neid}%" for i, neid in enumerate(selected_neids)}
    #     return self.fetch_data(query, params=params)

    # def get_vswr_data(self, selected_sites, end_date):
    #     like_conditions = " OR ".join(
    #         [f'"NE_NAME" LIKE :site_{i}' for i in range(len(selected_sites))]
    #     )

    #     start_date = end_date - pd.Timedelta(days=3)

    #     query = text(
    #         f"""
    #     SELECT
    #         "DATE_ID",
    #         "NE_NAME",
    #         "RRU",
    #         "pmReturnLossAvg",
    #         "VSWR"
    #     FROM ltevswr
    #     WHERE ({like_conditions})
    #     AND "DATE_ID" BETWEEN :start_date AND :end_date
    #     AND "RRU" NOT LIKE '%RfPort=R%'
    #     AND "RRU" NOT LIKE '%RfPort=S%'
    #     AND "VSWR" != 0
    #     """
    #     )
    #     params = {f"site_{i}": f"%{site}%" for i, site in enumerate(selected_sites)}
    #     params.update({"start_date": start_date, "end_date": end_date})

    #     return self.fetch_data(query, params=params)

    # def get_busyhour(self, selected_sites, end_date):
    #     like_conditions = " OR ".join(
    #         [f'"EUtranCellFDD" LIKE :site_{i}' for i in range(len(selected_sites))]
    #     )
    #     start_date = end_date - pd.Timedelta(days=15)
    #     query = text(
    #         f"""
    #     SELECT
    #         "DATE_ID",
    #         "EUtranCellFDD",
    #         "CQI"
    #     FROM ltebusyhour
    #     WHERE ({like_conditions})
    #     AND "DATE_ID" BETWEEN :start_date AND :end_date
    #     """
    #     )
    #     params = {f"site_{i}": f"%{site}%" for i, site in enumerate(selected_sites)}
    #     params.update({"start_date": start_date, "end_date": end_date})

    #     return self.fetch_data(query, params=params)

    # def get_cqi_cluster(self, eutrancellfdd, start_date, end_date):
    #     query = text(
    #         """
    #         SELECT
    #             "DATE_ID",
    #             "EUtranCellFDD",
    #             "CQI"
    #         FROM ltebusyhour
    #         WHERE "EUtranCellFDD" LIKE :eutrancellfdd
    #         AND "DATE_ID" BETWEEN :start_date AND :end_date
    #         """
    #     )
    #     params = {
    #         "eutrancellfdd": eutrancellfdd,
    #         "start_date": start_date,
    #         "end_date": end_date,
    #     }
    #     try:
    #         return pd.read_sql(query, self.engine, params=params)
    #     except Exception as e:
    #         st.error(f"Error fetching data: {e}")
    #         return pd.DataFrame()


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

    def create_charts_vswr(self, df, x1_param, x2_param, y_param, nename):
        # Calculate the average y_param for each combination of x1_param and x2_param
        avg_df = df.groupby([x1_param, x2_param])[y_param].mean().reset_index()

        unique_values = avg_df[x2_param].unique()
        colors = self.get_colors(len(unique_values))
        color_mapping = {cell: color for cell, color in zip(unique_values, colors)}

        fig = go.Figure()

        for value in unique_values:
            filtered_df = avg_df[avg_df[x2_param] == value]
            fig.add_trace(
                go.Bar(
                    x=filtered_df[x1_param],
                    y=filtered_df[y_param],
                    name=value,
                    marker_color=color_mapping[value],
                    hovertemplate=(
                        f"<b>®️ {value}</b><br>"
                        f"<b>{y_param}</b> 🟰  %{{y}}<br>"
                        f"<b>{x1_param}</b> 🟰  %{{x}}<br>"
                        "<extra></extra>"
                    ),
                    hoverlabel=dict(font_size=16, font_family="Vodafone"),
                )
            )

            fig.add_hline(
                y=1.3,
                line_dash="dashdot",
                line_color="#F70000",
                line_width=2,
            )

        fig.update_layout(
            barmode="group",
            yaxis_title=y_param,
            plot_bgcolor="#F5F5F5",
            paper_bgcolor="#F5F5F5",
            height=350,
            font=dict(family="Vodafone", size=18, color="#717577"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=-0.1,
                bgcolor="#F5F5F5",
                bordercolor="#FFFFFF",
                borderwidth=0,
                itemclick="toggleothers",
                itemdoubleclick="toggle",
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        container = st.container()
        with container:
            st.plotly_chart(fig, use_container_width=True)

    def cqiclusterchart(self, df, tier_data, xrule=None):
        df = df.rename(
            columns={"DATE_ID": "date", "EUtranCellFDD": "cellname", "CQI": "cqi"}
        )

        df["date"] = pd.to_datetime(df["date"])
        unique_cellnames = tier_data["cellname"].unique()
        unique_adjcellnames = tier_data["adjcellname"].unique()
        all_unique_names = sorted(
            np.unique(np.concatenate((unique_cellnames, unique_adjcellnames)))
        )
        colors = self.get_colors(len(all_unique_names))
        color_mapping = {name: color for name, color in zip(all_unique_names, colors)}
        fig = make_subplots(rows=1, cols=len(unique_cellnames), shared_yaxes=True)

        for i, cellname in enumerate(unique_cellnames):
            baseline_chart1 = df[df["cellname"] == cellname]
            baseline_chart1 = baseline_chart1.sort_values("date")
            color = color_mapping.get(cellname, "black")

            fig.add_trace(
                go.Scatter(
                    x=baseline_chart1["date"],
                    y=baseline_chart1["cqi"],
                    mode="lines+text",
                    line=dict(color=color, dash="dashdot", width=5),
                    name=f"{cellname}",
                    hovertemplate=(
                        f"<b>{cellname}</b><br><b>CQI :</b> %{{y}}<br><extra></extra>"
                    ),
                    hoverlabel=dict(bgcolor="white", font=dict(color="black")),
                ),
                row=1,
                col=i + 1,
            )

            adjcellnames = tier_data[tier_data["cellname"] == cellname][
                "adjcellname"
            ].unique()

            for adjcellname in adjcellnames:
                # Filter df for the current adjcellname
                tier_value = df[df["cellname"] == adjcellname]

                # Sort tier_value by date
                tier_value = tier_value.sort_values("date")

                # Get color for the current adjcellname from the color mapping
                color = color_mapping.get(
                    adjcellname, "black"
                )  # Default to black if not found

                # Create line chart for adjcellname
                fig.add_trace(
                    go.Scatter(
                        x=tier_value["date"],
                        y=tier_value["cqi"],
                        mode="lines+text",
                        line=dict(color=color),
                        name=f"{cellname} - {adjcellname}",
                        hovertemplate=(
                            f"<b>{adjcellname} 🔀 {cellname}</b><br>"
                            f"<b>CQI :</b> %{{y}}<br>"
                            "<extra></extra>"
                        ),
                        hoverlabel=dict(bgcolor="white", font=dict(color="black")),
                        textposition="top center",
                        textfont=dict(size=10, color=color),
                    ),
                    row=1,
                    col=i + 1,
                )

            if xrule:
                fig.add_vline(
                    x=st.session_state["xrule"],
                    line_width=2,
                    line_dash="dash",
                    row=1,
                    col=i + 1,
                )
        fig.update_layout(
            plot_bgcolor="#F5F5F5",
            paper_bgcolor="#F5F5F5",
            margin=dict(l=20, r=20, t=40, b=20),
            hoverlabel=dict(font_size=16, font_family="Plus Jakarta Sans"),
            hovermode="x unified",
            height=350,
            showlegend=True,
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

        with stylable_container(
            key="container_with_border",
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
                st.plotly_chart(fig, use_container_width=True)

    def create_charts_tastate(self, df):
        plot_columns = [
            "perc_300",
            "perc_500",
            "perc_700",
            "perc_1000",
            "perc_1500",
            "perc_2000",
            "perc_3000",
            "perc_5000",
            "perc_10000",
            "perc_15000",
            "perc_30000",
        ]

        if "ci" in df.columns:
            df = df[["ci", "Cell_Name"] + plot_columns]
            color_by = "ci"
        else:
            df = df[["Cell_Name"] + plot_columns]
            color_by = "Cell_Name"

        x_values = plot_columns
        unique_values = df[color_by].unique()
        colors = self.get_colors(len(unique_values))
        color_mapping = {value: color for value, color in zip(unique_values, colors)}

        fig = go.Figure()

        for value in unique_values:
            filtered_df = df[df[color_by] == value]
            cell_name = filtered_df["Cell_Name"].iloc[0]
            fig.add_trace(
                go.Bar(
                    x=x_values,
                    y=filtered_df.loc[:, x_values].values[0],
                    name=cell_name,
                    marker_color=color_mapping[value],
                    hovertemplate=(
                        f"<b>®️ {cell_name}</b><br>"
                        f"<b></b> %{{x}} - %{{y}}<br>"
                        "<extra></extra>"
                    ),
                    hoverlabel=dict(font_size=16, font_family="Vodafone"),
                )
            )

        fig.update_layout(
            barmode="group",
            xaxis_title=None,
            yaxis_title=None,
            plot_bgcolor="#F5F5F5",
            paper_bgcolor="#F5F5F5",
            height=500,
            font=dict(family="Vodafone", size=25, color="#717577"),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                bgcolor="#F5F5F5",
                bordercolor="#F5F5F5",
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                itemsizing="constant",
                font=dict(size=14),
            ),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        with stylable_container(
            key="container_with_border",
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
                st.plotly_chart(fig, use_container_width=True)

    def create_charts_tastate(self, df, sector):
        plot_columns = [
            "perc_300",
            "perc_500",
            "perc_700",
            "perc_1000",
            "perc_1500",
            "perc_2000",
            "perc_3000",
            "perc_5000",
            "perc_10000",
            "perc_15000",
            "perc_30000",
        ]

        if "ci" in df.columns:
            df = df[["ci", sector] + plot_columns]
            color_by = "ci"
        else:
            df = df[[sector] + plot_columns]
            color_by = sector

        x_values = plot_columns
        unique_values = df[color_by].unique()
        colors = self.get_colors(len(unique_values))
        color_mapping = {value: color for value, color in zip(unique_values, colors)}

        # Calculate grid dimensions
        total_sectors = len(df[sector].unique())
        max_cols = 6
        num_cols = min(max_cols, total_sectors)
        num_rows = math.ceil(total_sectors / num_cols)

        # Create subplots
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=df[sector].unique(),
            vertical_spacing=0.15,
            horizontal_spacing=0.05,
        )

        # Find global y-axis range
        y_min = df[plot_columns].min().min()
        y_max = df[plot_columns].max().max()
        y_range = [y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1]

        for i, sector_value in enumerate(df[sector].unique(), start=1):
            row = (i - 1) // num_cols + 1
            col = (i - 1) % num_cols + 1

            sector_df = df[df[sector] == sector_value]

            for value in sector_df[color_by].unique():
                filtered_df = sector_df[sector_df[color_by] == value]
                sector_name = filtered_df[sector].iloc[0]
                fig.add_trace(
                    go.Bar(
                        x=x_values,
                        y=filtered_df.loc[:, x_values].values[0],
                        name=sector_name,
                        marker_color=color_mapping[value],
                        hovertemplate=(
                            f"<b>®️ {sector_name}</b><br>"
                            f"<b></b> %{{x}} - %{{y}}<br>"
                            "<extra></extra>"
                        ),
                        hoverlabel=dict(font_size=16, font_family="Vodafone"),
                    ),
                    row=row,
                    col=col,
                )

            # Update axes for each subplot
            fig.update_xaxes(
                title_text="", tickangle=90, tickfont=dict(size=12), row=row, col=col
            )
            fig.update_yaxes(
                title_text="", tickfont=dict(size=12), range=y_range, row=row, col=col
            )

        # Update layout
        fig.update_layout(
            barmode="group",
            plot_bgcolor="#F5F5F5",
            paper_bgcolor="#F5F5F5",
            height=300 * num_rows,
            width=400 * num_cols,
            font=dict(family="Vodafone", size=12, color="#717577"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
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
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,  # Hide the legend as it might be too crowded
        )

        # Update subplot titles
        for i in fig["layout"]["annotations"]:
            i["font"] = dict(size=14)

        with stylable_container(
            key="container_with_border",
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
                st.plotly_chart(fig, use_container_width=True)

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

                        if yline:
                            yline_value = sector_data[yline].mean()
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
                            height=350,
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
                                    size=14,  # Updated size for x-axis tick labels
                                    color="#000000",
                                ),
                            ),
                        )

                        st.plotly_chart(fig, use_container_width=True)

    def create_charts_for_daily_reverse(
        self, df, cell_name, x_param, y_param, xrule=False, yline=None, y_range=None
    ):
        df = df.sort_values(by=x_param)
        df["sector"] = df[cell_name].apply(self.determine_sector)

        df[y_param] = df[y_param].astype(float)
        df = df[df[y_param] != 0]

        color_mapping = {
            cell: color
            for cell, color in zip(
                sorted(df[cell_name].unique()),
                self.get_colors(len(df[cell_name].unique())),
            )
        }

        sector_count = df["sector"].nunique()
        cols = min(sector_count, 3)
        columns = st.columns(cols)

        for idx, sector in enumerate(sorted(df["sector"].unique())):
            sector_data = df[df["sector"] == sector]

            y_min = sector_data[y_param].min()
            y_max = sector_data[y_param].max()

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
                                        f"<b>MC Class:</b> %{{customdata[0]}}<br>"
                                        f"<b>Band:</b> %{{customdata[1]}}<br>"
                                        f"<b>City:</b> %{{customdata[2]}}<br>"
                                        "<extra></extra>"
                                    ),
                                    customdata=np.stack(
                                        (
                                            cell_data["MC Class"],
                                            cell_data["Band"],
                                            cell_data["City"],
                                        ),
                                        axis=-1,
                                    ),
                                )
                            )

                        if yline:
                            yline_value = sector_data[yline].mean()
                            fig.add_hline(
                                y=yline_value,
                                line_dash="dashdot",
                                line_color="#F70000",
                                line_width=2,
                            )
                            # Adjust y_max if yline_value is greater
                            y_max = max(yline_value, y_max)

                        if xrule:
                            fig.add_vline(
                                x=st.session_state["xrule"],
                                line_width=2,
                                line_dash="dash",
                                line_color="#808080",
                            )

                        yaxis_range = [y_max, y_min]

                        fig.update_layout(
                            margin=dict(t=20, l=20, r=20, b=20),
                            title_text=f"SECTOR {sector}",
                            title_x=0.4,
                            template="plotly_white",
                            hoverlabel=dict(font_size=16, font_family="Vodafone"),
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
                            ),
                            paper_bgcolor="#F5F5F5",
                            plot_bgcolor="#F5F5F5",
                            width=600,
                            height=350,
                            showlegend=True,
                            xaxis=dict(
                                tickfont=dict(
                                    size=14,
                                    color="#000000",
                                ),
                            ),
                            yaxis=dict(
                                autorange="reversed",
                                range=yaxis_range,
                                tickfont=dict(size=14, color="#000000"),
                            ),
                        )

                        st.plotly_chart(fig, use_container_width=True)

    def create_charts_for_stacked_area(
        self, df, cell_name, x_param, y_param, xrule=False, yline=None, y_range=None
    ):
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
                        if yline:
                            yline_value = sector_data[yline].mean()
                            fig.add_hline(
                                y=yline_value,
                                line_dash="dot",
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
                            title_text=f"SECTOR {sector}",
                            title_x=0.4,
                            xaxis_title=None,
                            yaxis_title=None,
                            template="plotly_white",
                            hoverlabel=dict(font_size=16, font_family="Vodafone"),
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
                            height=350,
                            showlegend=True,
                        )

                        st.plotly_chart(fig, use_container_width=True)

    def create_charts_for_stacked_area_neid(
        self, df, neid, x_param, y_param, xrule=False
    ):
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
                # template="plotly_white",
                hoverlabel=dict(font_size=16, font_family="Vodafone"),
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
                        size=14,  # Updated size for y-axis tick labels
                        color="#000000",
                    ),
                ),
                xaxis=dict(
                    tickfont=dict(
                        size=14,  # Updated size for x-axis tick labels
                        color="#000000",
                    ),
                ),
                paper_bgcolor="#F5F5F5",
                plot_bgcolor="#F5F5F5",
                width=600,
                height=350,
                showlegend=True,
            )

            if xrule:
                fig.add_vline(
                    x=st.session_state["xrule"],
                    line_width=2,
                    line_dash="dash",
                    line_color="#808080",
                )

            st.plotly_chart(fig, use_container_width=True)

    def create_charts_for_mulsec(self, df, cell_name, x_param, y_param, y_param2):
        try:
            # Check if 'mulsec_category' column exists
            if "mulsec_category" not in df.columns:
                st.error("The site does not have a multisector.")
                return

            # Check if all values in 'mulsec_category' are None
            if df["mulsec_category"].isna().all():
                st.error("The site does not have a multisector.")
                return

            # Filter out rows where mulsec_category is None
            df = df[df["mulsec_category"].notna()]

            # Sort and convert the y_param and y_param2 columns to float
            df = df.sort_values(by=x_param)
            df[y_param] = df[y_param].astype(float)
            df[y_param2] = df[y_param2].astype(float)

            unique_cells = df[cell_name].unique().tolist()

            color_mapping = {
                cell: color
                for cell, color in zip(
                    sorted(
                        df[cell_name].unique()
                    ),  # Sort the unique values of cell_name
                    self.get_colors(len(df[cell_name].unique())),
                )
            }

            mulsec_groups = df.groupby("mulsec_category")
            num_groups = len(mulsec_groups)

            # Determine the number of columns
            cols = num_groups
            rows = 2

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[
                    f"{group} - PRB Utilization" for group, _ in mulsec_groups
                ]
                + [f"{group} - Active Users" for group, _ in mulsec_groups],
                shared_xaxes=True,
                vertical_spacing=0.1,
                horizontal_spacing=0.03,
            )

            for idx, (mulsec, group) in enumerate(mulsec_groups):
                col = idx + 1

                for cell in group[cell_name].unique():
                    cell_data = group[group[cell_name] == cell]
                    color = color_mapping[cell]

                    # Plot y_param in the first row
                    fig.add_trace(
                        go.Scatter(
                            x=cell_data[x_param],
                            y=cell_data[y_param],
                            mode="lines",
                            name=cell,
                            line=dict(color=color, width=2),
                            legendgroup=cell,
                            hovertemplate=(
                                f"<b>{cell}</b><br>"
                                f"<b>%{{x}}</b><br>"
                                f"<b>PRB:</b> %{{y}}<br>"
                                "<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=col,
                    )

                    # Plot y_param2 in the second row
                    fig.add_trace(
                        go.Scatter(
                            x=cell_data[x_param],
                            y=cell_data[y_param2],
                            mode="lines",
                            name=f"{cell} ({y_param2})",
                            line=dict(color=color, width=2),
                            legendgroup=cell,
                            showlegend=False,
                            hovertemplate=(
                                f"<b>{cell}</b><br>"
                                f"<b>%{{x}}</b><br>"
                                f"<b>PRB:</b> %{{y}}<br>"
                                "<extra></extra>"
                            ),
                        ),
                        row=2,
                        col=col,
                    )

            fig.update_layout(
                showlegend=True,
                height=600,
                template="plotly_white",
                # hovermode="x unified",
                margin=dict(l=20, r=20, t=20, b=5),
                hoverlabel=dict(font_size=16, font_family="Vodafone"),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.1,
                    xanchor="center",
                    x=0.5,
                    itemclick="toggleothers",
                    itemdoubleclick="toggle",
                    itemsizing="constant",
                    font=dict(size=14),
                    title=None,
                ),
                paper_bgcolor="#F5F5F5",
                plot_bgcolor="#F5F5F5",
            )

            with stylable_container(
                key="container_with_border",
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
                    st.plotly_chart(fig, use_container_width=True)

        except KeyError as e:
            st.error(f"KeyError: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")


class App:
    def __init__(self):
        self.config = Config().load()
        self.database_session = DatabaseSession(self.config)
        self.query_manager = None
        self.dataframe_manager = DataFrameManager()
        self.streamlit_interface = StreamlitInterface()
        self.chart_generator = ChartGenerator()
        self.geodata = None

    def run(self):
        session, engine = self.database_session.create_session()
        if session is None:
            return

        self.query_manager = QueryManager(engine)

        script_dir = os.path.dirname(__file__)

        sitelist_path = os.path.join(script_dir, "test_sitelist.csv")
        sitelist = self.streamlit_interface.load_sitelist(sitelist_path)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        assets_image = os.path.join(project_root, "assets/")

        # MARK: apply mulsec logic to dataframe
        st.cache_data(ttl=1200)

        eutran_mapping = {
            "L23_F2_S1": ["MF1", "MF01", "IF01", "HF01", "OF01"],
            "L23_F1_S2": ["ME2", "ME02"],
            "L23_F1_S1": ["ME1", "ME01"],
            "L23_F2_S2": ["MF2", "MF02"],
            "L23_F1_S3": ["ME3", "ME03"],
            "L23_F2_S3": ["MF3", "MF03"],
            "L23_F1_S4": ["ME4", "ME04"],
            "L23_F2_S4": ["MF4", "MF04"],
            "L23_F1_S5": ["ME5", "ME05"],
            "L23_F2_S5": ["MF5", "MF05"],
            "L23_F1_S6": ["ME6", "ME06"],
            "L23_F2_S6": ["MF6", "MF06"],
            "L23_F3_S1": ["MI1", "MI01"],
            "L23_F3_S2": ["MI2", "MI02"],
            "L23_F3_S3": ["MI3", "MI03"],
            "L23_MIMO_50_F3_S1": ["MV01"],
            "L23_MIMO_50_F3_S2": ["MV02"],
            "L23_MIMO_50_F3_S3": ["MV03"],
            "L23_MIMO_F1_S1": ["VE01"],
            "L23_MIMO_F1_S2": ["VE02"],
            "L23_MIMO_F1_S3": ["VE03"],
            "L23_MIMO_F2_S1": ["VF01"],
            "L23_MIMO_F2_S2": ["VF02"],
            "L23_MIMO_F2_S3": ["VF03"],
            "L23_MIMO_F3_S1": ["VV01"],
            "L23_MIMO_F3_S2": ["VV02"],
            "L23_MIMO_F3_S3": ["VV03"],
            "L9_S1": ["MT1", "MT01"],
            "L9_S2": ["MT2", "MT02"],
            "L9_S3": ["MT3", "MT03"],
            "L9_S4": ["MT4", "MT04"],
            "L9_S5": ["MT5", "MT05"],
            "L9_S6": ["MT6", "MT06"],
            "L18_S1": ["ML1", "ML01"],
            "L18_S2": ["ML2", "ML02"],
            "L18_S3": ["ML3", "ML03"],
            "L18_S4": ["ML4", "ML04"],
            "L18_S5": ["ML5", "ML05"],
            "L18_S6": ["ML6", "ML06"],
            "L21_S1": ["MR1", "MR01"],
            "L21_S2": ["MR2", "MR02"],
            "L21_S3": ["MR3", "MR03"],
            "L21_S4": ["MR4", "MR04"],
            "L21_S5": ["MR5", "MR05"],
            "L21_S6": ["MR6", "MR06"],
        }

        reverse_mapping = {v: k for k, vals in eutran_mapping.items() for v in vals}

        def find_mapping(value):
            for suffix in reverse_mapping:
                if value.endswith(suffix):
                    return reverse_mapping[suffix]
            return "Unknown"

        def split_sector(df, col, new_col="eutrancell_new"):
            # Apply the find_mapping function to each row
            df[new_col] = df[col].apply(find_mapping)
            return df

        def add_mulsec_category(df):
            def get_mulsec_category(cell, cells):
                try:
                    cell = cell.upper()
                    cells_upper = [c.upper() for c in cells]

                    if cell.endswith(("MT04", "MT4")) and any(
                        c.endswith(("MT01", "MT1")) or c.endswith(("TT01", "TT02"))
                        for c in cells_upper
                    ):
                        return "MULSEC SEC 1", ("MT01", "MT1")
                    elif cell.endswith(("MT05", "MT5")) and any(
                        c.endswith(("MT02", "MT2")) for c in cells_upper
                    ):
                        return "MULSEC SEC 2", ("MT02", "MT2")
                    elif cell.endswith(("MT06", "MT6")) and any(
                        c.endswith(("MT03", "MT3")) for c in cells_upper
                    ):
                        return "MULSEC SEC 3", ("MT03", "MT3")
                    elif cell.endswith(("ML04", "ML4")) and any(
                        c.endswith(("ML01", "ML1")) for c in cells_upper
                    ):
                        return "MULSEC SEC 1", ("ML01", "ML1")
                    elif cell.endswith(("ML05", "ML5")) and any(
                        c.endswith(("ML02", "ML2")) for c in cells_upper
                    ):
                        return "MULSEC SEC 2", ("ML02", "ML2")
                    elif cell.endswith(("ML06", "ML6")) and any(
                        c.endswith(("ML03", "ML3")) for c in cells_upper
                    ):
                        return "MULSEC SEC 3", ("ML03", "ML3")
                    elif cell.endswith(("MR04", "MR4")) and any(
                        c.endswith(("MR01", "MR1")) for c in cells_upper
                    ):
                        return "MULSEC SEC 1", ("MR01", "MR1")
                    elif cell.endswith(("MR05", "MR5")) and any(
                        c.endswith(("MR02", "MR2")) for c in cells_upper
                    ):
                        return "MULSEC SEC 2", ("MR02", "MR2")
                    elif cell.endswith(("MR06", "MR6")) and any(
                        c.endswith(("MR03", "MR3")) for c in cells_upper
                    ):
                        return "MULSEC SEC 3", ("MR03", "MR3")
                    else:
                        return None, None
                except Exception as e:
                    print(f"Error in get_mulsec_category: {e}")
                    return None, None

            def find_partner_and_assign(df, cell, category, partner_suffixes):
                try:
                    cell_upper = cell.upper()
                    partner_cells = [
                        c
                        for c in df["EUtranCellFDD"]
                        if any(
                            c.upper().endswith(suffix) for suffix in partner_suffixes
                        )
                    ]
                    if partner_cells:
                        df.loc[
                            df["EUtranCellFDD"].str.upper() == cell_upper,
                            "mulsec_category",
                        ] = category
                        df.loc[
                            df["EUtranCellFDD"]
                            .str.upper()
                            .isin([p.upper() for p in partner_cells]),
                            "mulsec_category",
                        ] = category
                except KeyError as e:
                    print(f"KeyError in find_partner_and_assign: {e}")
                except Exception as e:
                    print(f"Error in find_partner_and_assign: {e}")

            try:
                # Ensure 'mulsec_category' column exists
                if "mulsec_category" not in df.columns:
                    df["mulsec_category"] = None

                # First pass to determine all possible mulsec categories
                for cell in df["EUtranCellFDD"]:
                    category, partner_suffixes = get_mulsec_category(
                        cell, df["EUtranCellFDD"]
                    )
                    if category:
                        find_partner_and_assign(df, cell, category, partner_suffixes)
            except KeyError as e:
                print(f"KeyError in add_mulsec_category: {e}")
            except Exception as e:
                print(f"Error in add_mulsec_category: {e}")

            return df

        (
            col1,
            col2,
            col3,
            col4,
            _,
        ) = st.columns([1, 1, 1, 1, 3])
        with col1:
            date_range = self.streamlit_interface.select_date_range()
            st.session_state["date_range"] = date_range

        with col2:
            selected_sites = self.streamlit_interface.site_selection(sitelist)
            st.session_state.selected_sites = selected_sites

        with col3:
            selected_neids = self.streamlit_interface.neid_selection(sitelist)
            st.session_state.selected_neids = selected_neids

        with col4:
            xrule = self.streamlit_interface.select_xrule_date()
            st.session_state["xrule"] = xrule

        # Fetch data for each selected site
        if st.button("Run Query"):
            if selected_sites and date_range:
                start_date, end_date = date_range
                combined_target_data = []
                combined_ltetastate_data = []
                combined_ltedaily_data = []

                # Fetch MCOM data
                def load_tier_data(tier_path):
                    try:
                        tier_data = pd.read_csv(tier_path)
                    except FileNotFoundError:
                        st.error(f"File does not exist: {tier_path}")
                        tier_data = None
                    except pd.errors.EmptyDataError:
                        st.write(f"The file {tier_path} is empty.")
                        tier_data = None
                    except Exception as e:
                        st.write(f"An error occurred while reading {tier_path}: {e!s}")
                        tier_data = None
                    return tier_data

                def load_isd_data(isd_path):
                    try:
                        isd_data = pd.read_csv(isd_path)
                    except FileNotFoundError:
                        st.error(f"File does not exist: {isd_path}")
                        isd_data = None
                    except pd.errors.EmptyDataError:
                        st.write(f"The file {isd_path} is empty.")
                        isd_data = None
                    except Exception as e:
                        st.write(f"An error occurred while reading {isd_path}: {e!s}")
                        isd_data = None
                    return isd_data

                def calculate_rf(df_isd_data, df_tastate_data):
                    try:
                        df_isd_data["ta_overshoot"] = df_isd_data["isd"] + (
                            0.1 * df_isd_data["isd"]
                        )
                        df_isd_data["ta_undershoot"] = df_isd_data["isd"] / 3
                        df_isd_data["ta_overlap"] = df_isd_data["isd"] / 2 + (
                            0.2 * df_isd_data["isd"]
                        )

                        df_merged = pd.merge(
                            df_isd_data,
                            df_tastate_data,
                            left_on=["ci", "enbid"],
                            right_on=["ci", "enodebid"],
                        )

                        conditions = [
                            (df_merged["isd"] > 5),
                            (df_merged["perc90_ta_distance_km"] > df_merged["isd"]),
                            (
                                df_merged["perc90_ta_distance_km"]
                                < 0.5 * df_merged["isd"]
                            ),
                            (
                                df_merged["perc90_ta_distance_km"]
                                > 0.5 * df_merged["ta_overlap"]
                            ),
                            (
                                df_merged["perc90_ta_distance_km"]
                                == 0.5 * df_merged["ta_overlap"]
                            ),
                        ]

                        choices = [
                            "📐 Open Area",
                            "📈 TA Overshoot",
                            "📉 TA Undershoot",
                            "❌ TA Overlap",
                            "✔️ TA Proper",
                        ]

                        df_merged["final_ta_status"] = np.select(
                            conditions, choices, default="Open Area"
                        )

                        # Select final columns
                        final_df = df_merged[
                            [
                                "siteid",
                                "neid",
                                "eutrancell",
                                "isd",
                                "ta_overshoot",
                                "ta_undershoot",
                                "ta_overlap",
                                "final_ta_status",
                            ]
                        ]

                        return final_df

                    except Exception as e:
                        print(f"An error occurred: {e}")
                        return None

                # Define a function to display images or messages
                def display_image_or_message(
                    column, folder_path, image_name, message_prefix
                ):
                    image_path = os.path.join(folder_path, image_name)
                    if os.path.exists(image_path):
                        column.image(image_path, caption=None, use_column_width=True)
                    else:
                        column.write(f"{message_prefix}: {image_path}")

                # Define a function to style and display markdown
                def display_styled_markdown(
                    column, text, font_size=24, text_align="left", tag="h6"
                ):
                    styling_args = {
                        "font_size": font_size,
                        "text_align": text_align,
                        "tag": tag,
                    }
                    column.markdown(*styling(text, **styling_args))

                for siteid in selected_sites:
                    folder = os.path.join(project_root, "sites", siteid)
                    tier_path = os.path.join(folder, "tier.csv")
                    isd_path = os.path.join(folder, "isd.csv")

                    if os.path.exists(folder):
                        tier_data = load_tier_data(tier_path)
                    else:
                        st.error(f"Folder does not exist: {folder}")
                        tier_data = None

                    if os.path.exists(folder):
                        isd_data = load_isd_data(isd_path)
                    else:
                        st.error(f"Folder does not exist: {folder}")
                        isd_data = None

                    # st.write(f"ISD data: {isd_data}")

                    if tier_data is not None:
                        unique_eutrancellfdd = sorted(
                            set(tier_data["cellname"].unique()).union(
                                set(tier_data["adjcellname"].unique())
                            )
                        )
                        all_data = pd.concat(
                            [
                                self.query_manager.get_cqi_cluster(
                                    eutrancellfdd, start_date, end_date
                                )
                                for eutrancellfdd in unique_eutrancellfdd
                            ],
                            ignore_index=True,
                        )

                        self.dataframe_manager.add_dataframe(
                            f"cqitier_{siteid}", all_data
                        )

                    sac.divider(color="black", align="center")
                    col1, col2, _ = st.columns([1, 1, 5])

                    with col1.container():
                        with stylable_container(
                            key="erilogo",
                            css_styles="""
                                img {
                                    display: block;
                                    margin-left: auto;
                                    margin-right: auto;
                                    width: 100%;
                                    position: relative;
                                    top: 5px;
                                }
                            """,
                        ):
                            st.image(assets_image + "eri.png")

                    with col2.container():
                        with stylable_container(
                            key="tsellogo",
                            css_styles="""
                                img {
                                    display: block;
                                    margin-left: auto;
                                    margin-right: auto;
                                    width: 100%;
                                    position: relative;
                                    top: 0px;  /* Adjust this value as needed */
                                }
                            """,
                        ):
                            st.image(assets_image + "tsel.png")
                    st.markdown("# ")
                    col1, col2 = st.columns([2, 1])

                    # Display styled markdowns
                    display_styled_markdown(col1, f"📝 Naura Site {siteid}")
                    display_styled_markdown(col2, f"⚠️ Alarm Site {siteid}")

                    # Create containers with borders
                    con1 = col1.container(border=True)
                    con2 = col2.container(border=True)

                    if os.path.exists(folder):
                        display_image_or_message(
                            con1, folder, "naura.jpg", "Please upload the image"
                        )
                        display_image_or_message(
                            con2, folder, "alarm.jpg", "Please upload the image"
                        )
                    else:
                        st.error(f"Path does not exist: {folder}")

                    mcom_data = self.query_manager.get_mcom_data(siteid)
                    self.dataframe_manager.add_dataframe(
                        f"mcom_data_{siteid}", mcom_data
                    )
                    # st.table(mcom_data)

                    for _, row in mcom_data.iterrows():
                        target_data = self.query_manager.get_target_data(
                            row["KABUPATEN"],
                            # row["MC_class"],
                            row["LTE"],
                        )
                        target_data["EutranCell"] = row["Cell_Name"]
                        combined_target_data.append(target_data)

                    # Fetch LTE daily data for each site
                    ltedaily_data = self.query_manager.get_ltedaily_data(
                        siteid, selected_neids, start_date, end_date
                    )
                    combined_ltedaily_data.append(ltedaily_data)
                    self.dataframe_manager.add_dataframe(
                        f"ltedaily_data_{siteid}", ltedaily_data
                    )

                # Combine target data and display
                if combined_target_data:
                    combined_target_df = pd.concat(
                        combined_target_data, ignore_index=True
                    )
                    self.dataframe_manager.add_dataframe(
                        "combined_target_data", combined_target_df
                    )
                    self.dataframe_manager.display_dataframe(
                        "combined_target_df", "target"
                    )

                # Combine ltetastate data and display
                if combined_ltetastate_data:
                    combined_ltetastate_df = pd.concat(
                        combined_ltetastate_data, ignore_index=True
                    )
                    self.dataframe_manager.add_dataframe(
                        "combined_ltetastate_data", combined_ltetastate_df
                    )

                # Combine ltedaily data and display
                if combined_ltedaily_data:
                    combined_ltedaily_df = pd.concat(
                        combined_ltedaily_data, ignore_index=True
                    )
                    self.dataframe_manager.add_dataframe(
                        "combined_ltedaily_data", combined_ltedaily_df
                    )

                    # Combine target data with ltedaily data
                    combined_target_ltedaily_df = pd.merge(
                        combined_target_df,
                        combined_ltedaily_df,
                        on="EutranCell",
                        how="right",
                    )
                    # st.table(combined_target_ltedaily_df)
                    final_data_ltedaily = split_sector(
                        combined_target_ltedaily_df, "EutranCell"
                    )
                    # st.write(final_data_ltedaily)
                    # self.dataframe_manager.add_dataframe(
                    #     "combined_target_ltedaily_data", combined_target_ltedaily_df
                    # )

                    # FLAG: Start Charts
                    st.markdown(
                        *styling(
                            f"📶 Service Availability for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="Availability",
                        xrule=True,
                    )
                    st.markdown(
                        *styling(
                            f"📶 RRC Success Rate for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    # FLAG: - create_charts_for_daily
                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="RRC_SR",
                        yline="CSSR",
                        xrule=True,
                    )

                    # TAG: - ERAB SR
                    st.markdown(
                        *styling(
                            f"📶 E-RAB Setup Success Rate for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="ERAB_SR",
                        yline="CSSR",
                        xrule=True,
                    )

                    # TAG: - SSSR
                    st.markdown(
                        *styling(
                            f"📶 Session Setup Success Rate for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="SSSR",
                        yline="CSSR",
                        xrule=True,
                    )

                    # TAG: - CSSR
                    st.markdown(
                        *styling(
                            f"📶 Session Abnormal Release Sectoral for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="SAR",
                        yline="Service Drop Rate",
                        xrule=True,
                    )

                    # TAG: - CQI Non HOM
                    st.markdown(
                        *styling(
                            f"📶 CQI Distribution (Non-Homogeneous) Sectoral for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="avgcqinonhom",
                        yline="CQI",
                        xrule=True,
                    )

                    # TAG: - Spectral Efficiency
                    st.markdown(
                        *styling(
                            f"📶 Spectral Efficiency Analysis Sectoral for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="SE_DAILY",
                        yline="SE",
                        xrule=True,
                    )

                    # TAG: - Intra HO SR
                    st.markdown(
                        *styling(
                            f"📶 Intra-Frequency Handover Performance Sectoral for Site {siteid} (%)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="Intra_HO_Exe_SR",
                        yline="Intra Freq HOSR",
                        xrule=True,
                    )

                    # TAG: - Inter HO SR
                    st.markdown(
                        *styling(
                            f"📶 Inter-Frequency Handover Performance Sectoral for Site {siteid} (%)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="Inter_HO_Exe_SR",
                        yline="Inter Freq HOSR",
                        xrule=True,
                    )

                    # TAG: - UL RSSI
                    st.markdown(
                        *styling(
                            f"📶 Average UL RSSI Sectoral for Site {siteid} (dBm)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily_reverse(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="UL_INT_PUSCH_y",
                        yline="UL_INT_PUSCH_x",
                        xrule=True,
                    )

                    st.markdown(
                        *styling(
                            f"📶 Throughput (Mbps) Sectoral for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_daily(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="CellDownlinkAverageThroughput",
                        xrule=True,
                    )

                    st.markdown(
                        *styling(
                            f"📶 Payload Distribution Sectoral for Site {siteid} (Gpbs)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    self.chart_generator.create_charts_for_stacked_area(
                        df=final_data_ltedaily,
                        cell_name="eutrancell_new",
                        x_param="DATE_ID",
                        y_param="Payload_Total(Gb)",
                        xrule=True,
                    )

                payload_data = self.query_manager.get_ltedaily_payload(
                    selected_sites, start_date, end_date
                )
                self.dataframe_manager.add_dataframe("payload_data", payload_data)
                # st.write(payload_data)

                ltehourly_data = self.query_manager.get_ltehourly_data(
                    selected_sites, end_date
                )

                ltehourly_data["datetime"] = pd.to_datetime(
                    ltehourly_data["DATE_ID"].astype(str)
                    + " "
                    + ltehourly_data["hour_id"].astype(str).str.zfill(2),
                    format="%Y-%m-%d %H",
                )

                ltebusyhour_data = self.query_manager.get_busyhour(
                    selected_sites, end_date
                )
                # self.dataframe_manager.add_dataframe(
                #     "ltebusyhour_data", ltebusyhour_data
                # )
                ltebusyhour_data_final = split_sector(ltebusyhour_data, "EUtranCellFDD")

                self.dataframe_manager.add_dataframe("ltehourly_data", ltehourly_data)
                ltehourly_data_mulsec = add_mulsec_category(ltehourly_data)
                # st.write(ltehourly_data_mulsec)
                ltehourly_data_final = split_sector(
                    ltehourly_data_mulsec, "EUtranCellFDD"
                )
                # st.write(ltehourly_data_final)
                # st.write(ltehourly_data_final)
                vswr_data = self.query_manager.get_vswr_data(selected_sites, end_date)
                self.dataframe_manager.add_dataframe("vswr_data", vswr_data)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(
                        *styling(
                            f"📶 Payload Distribution by Frequency for Site  {siteid} (Gpbs)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                with col2:
                    st.markdown(
                        *styling(
                            f"📶 Total Site Payload Overview for Site {siteid} (Gpbs)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                # TAG: - Payload Neid
                col1, col2 = st.columns([1, 1])
                con1 = col1.container()
                con2 = col2.container()
                with con1:
                    with stylable_container(
                        key="container_with_border",
                        css_styles="""
                            {
                                background-color: #F5F5F5;
                                border: 2px solid rgba(49, 51, 63, 0.2);
                                border-radius: 0.5rem;
                                padding: calc(1em - 1px)
                            }
                            """,
                    ):
                        self.chart_generator.create_charts_for_stacked_area_neid(
                            df=payload_data,
                            neid="NEID",
                            x_param="DATE_ID",
                            y_param="Payload_Total(Gb)",
                            xrule=True,
                        )

                with con2:
                    with stylable_container(
                        key="container_with_border",
                        css_styles="""
                            {
                                background-color: #F5F5F5;
                                border: 2px solid rgba(49, 51, 63, 0.2);
                                border-radius: 0.5rem;
                                padding: calc(1em - 1px)
                            }
                            """,
                    ):
                        self.chart_generator.create_charts_for_stacked_area_neid(
                            df=payload_data,
                            neid="SITEID",
                            x_param="DATE_ID",
                            y_param="Payload_Total(Gb)",
                            xrule=True,
                        )

                st.markdown(
                    *styling(
                        f"📶 CQI Comparison Across Frequencies for Site  {siteid}",
                        font_size=24,
                        text_align="left",
                        tag="h6",
                    )
                )
                # st.write(ltebusyhour_data_final)
                self.chart_generator.create_charts_for_daily(
                    df=ltebusyhour_data_final,
                    cell_name="eutrancell_new",
                    x_param="DATE_ID",
                    y_param="CQI",
                    xrule=True,
                )

                # MARK: CQI 1st tier
                st.markdown(
                    *styling(
                        f"📶 CQI: {siteid} vs. Tier 1",
                        font_size=24,
                        text_align="left",
                        tag="h6",
                    )
                )
                if tier_data is not None:
                    try:
                        self.chart_generator.cqiclusterchart(
                            all_data,
                            tier_data,
                            xrule,
                        )
                    except Exception as e:
                        st.write(f"An error occurred: {e!s}")

                cols = st.columns(1)
                with cols[0]:
                    st.markdown(
                        *styling(
                            f"📶 PRB Utilization VS Active User Multisector for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                    # FLAG: transform to mulsec
                    self.chart_generator.create_charts_for_mulsec(
                        df=ltehourly_data_final,
                        cell_name="eutrancell_new",
                        x_param="datetime",
                        y_param="DL_Resource_Block_Utilizing_Rate",
                        y_param2="Active User",
                    )

                st.markdown(
                    *styling(
                        f"📶 PRB Utilization Site {siteid}",
                        font_size=24,
                        text_align="left",
                        tag="h6",
                    )
                )
                self.chart_generator.create_charts_for_daily(
                    df=ltehourly_data_final,
                    cell_name="eutrancell_new",
                    x_param="datetime",
                    y_param="DL_Resource_Block_Utilizing_Rate",
                )
                st.markdown(
                    *styling(
                        f"📶 Active User Site {siteid}",
                        font_size=24,
                        text_align="left",
                        tag="h6",
                    )
                )
                self.chart_generator.create_charts_for_daily(
                    df=ltehourly_data_final,
                    cell_name="eutrancell_new",
                    x_param="datetime",
                    y_param="Active User",
                )

                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(
                        *styling(
                            f"📶 VSWR Analysis for Site {siteid} (dBm)",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                with col2:
                    st.markdown(
                        *styling(
                            f"📶 RET After {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )

                col1, col2 = st.columns([3, 2])
                con1 = col1.container()
                con2 = col2.container()
                with con1:
                    with stylable_container(
                        key="container_with_border",
                        css_styles="""

                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            position: relative;
                            top: 0px;  /* Adjust this value as needed */
                        }
                        container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px)
                        }
                        """,
                    ):
                        self.chart_generator.create_charts_vswr(
                            df=vswr_data,
                            x1_param="DATE_ID",
                            x2_param="RRU",
                            y_param="VSWR",
                            nename="RRU",
                        )

                with con2:
                    with stylable_container(
                        key="container_with_border",
                        css_styles="""
                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            max-width: 80%;
                            position: relative;
                            top: 0px;
                        }
                        .custom-container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px);
                        }
                        """,
                    ):
                        if os.path.exists(folder):
                            ret = os.path.join(folder, "ret.jpg")
                            if os.path.exists(ret):
                                st.image(ret, caption=None, use_column_width=True)
                            else:
                                st.error(f"Please upload the image: {ret}")
                        else:
                            st.error(f"Path does not exist: {folder}")
                with con2:
                    st.markdown(
                        *styling(
                            f"📶 VSWR for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h4",
                        )
                    )
                    with stylable_container(
                        key="container_with_border",
                        css_styles="""
                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            max-width: 80%;
                            position: relative;
                            top: 0px;
                        }
                        .custom-container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px);
                        }
                        """,
                    ):
                        if os.path.exists(folder):
                            vswr = os.path.join(folder, "vswr.jpg")
                            if os.path.exists(vswr):
                                st.image(vswr, caption=None, use_column_width=True)
                            else:
                                st.error(f"Please upload the image: {vswr}")
                        else:
                            st.error(f"Path does not exist: {folder}")
                # MARK: - GeoApp MDT Data
                try:
                    mcom_data1 = self.query_manager.get_mcom_data(siteid)
                    st.session_state.mcom_data1 = mcom_data1
                    mcom_data = self.query_manager.get_mcom_data(siteid)
                    st.session_state.mcom_data = mcom_data
                    st.session_state.combined_target_data = combined_target_data

                    required_columns = [
                        "Site_ID",
                        "NODE_ID",
                        "NE_ID",
                        "Cell_Name",
                        "Longitude",
                        "Latitude",
                        "Dir",
                        "Ant_BW",
                        "Ant_Size",
                        "cellId",
                        "eNBId",
                        "KABUPATEN",
                        "LTE",
                    ]

                    try:
                        if not all(
                            column in mcom_data.columns for column in required_columns
                        ):
                            raise ValueError(
                                "The dataframe does not contain all required columns"
                            )

                        self.dataframe_manager.add_dataframe(
                            f"mcom_data_{siteid}", mcom_data
                        )

                        if isinstance(selected_neids, list):
                            selected_neids = selected_neids[0]

                        if isinstance(selected_neids, str):
                            ltemcomdata = mcom_data[
                                mcom_data["NE_ID"] == selected_neids
                            ]
                        else:
                            raise ValueError(
                                "selected_neid should be a string or a list containing a string"
                            )

                    except ValueError as ve:
                        st.error(f"An error occurred: {ve!s}")
                    except KeyError as ke:
                        st.error(f"A column is missing in the dataframe: {ke!s}")
                    except Exception as e:
                        st.error(
                            "An unexpected error occurred while processing the data."
                        )

                except Exception as e:
                    st.error(
                        "An error occurred while fetching data. Please try again later."
                    )

                # MARK: - GeoApp MDT Data
                ltemdtdata = self.query_manager.get_ltemdt_data(selected_sites)
                self.dataframe_manager.add_dataframe("ltemdtdata", ltemdtdata)

                ltemcomdata["eNBId"] = ltemcomdata["eNBId"].astype(float)
                ltemcomdata["cellId"] = ltemcomdata["cellId"].astype(float)
                ltemdtdata["enodebid"] = ltemdtdata["enodebid"].astype(float)
                ltemdtdata["ci"] = ltemdtdata["ci"].astype(float)

                filter_set = set(zip(ltemcomdata["eNBId"], ltemcomdata["cellId"]))

                ltemdtdata_final = ltemdtdata.loc[
                    ltemdtdata[["enodebid", "ci"]].apply(tuple, axis=1).isin(filter_set)
                ]

                ltetastate_data = self.query_manager.get_ltetastate_data(selected_sites)
                self.dataframe_manager.add_dataframe("ltetastate_data", ltetastate_data)

                mcom_data["cellId"] = mcom_data["cellId"].astype(float)
                mcom_data["eNBId"] = mcom_data["eNBId"].astype(float)
                ltetastate_data["ci"] = ltetastate_data["ci"].astype(float)
                ltetastate_data["enodebid"] = ltetastate_data["enodebid"].astype(float)

                mcom_ta_renamed = mcom_data.rename(
                    columns={"cellId": "ci", "eNBId": "enodebid"}
                )

                mcom_ta_indexed = mcom_ta_renamed.set_index(["ci", "enodebid"])
                ltetastate_data_indexed = ltetastate_data.set_index(["ci", "enodebid"])
                tastate_data = pd.merge(
                    mcom_ta_indexed,
                    ltetastate_data_indexed,
                    on=["ci", "enodebid"],
                    how="inner",
                )
                # tastate_final = split_sector(tastate_data, "Cell_Name")
                # MARK: use this to calculate RF Propagation
                self.dataframe_manager.add_dataframe("tastate_data", tastate_data)
                ta_state = split_sector(tastate_data, "Cell_Name")
                # st.write(ta_state)
                col1, col2 = st.columns([3, 2])
                con1 = col1.container()
                con2 = col2.container()
                # MARK: - GeoApp MDT Data
                st.session_state.mcom_data = mcom_data
                st.session_state.combined_target_data = combined_target_data

                # MARK: - GeoApp MDT Data
                ltemdtdata = self.query_manager.get_ltemdt_data(selected_sites)
                self.dataframe_manager.add_dataframe("ltemdtdata", ltemdtdata)
                st.session_state.ltemdtdata = ltemdtdata
                st.session_state.ltemdtdata_final = ltemdtdata_final
                st.session_state.ltemcomdata = ltemdtdata_final
                with con1:
                    st.markdown(
                        *styling(
                            f"☢️ MDT for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h6",
                        )
                    )
                    self.geodata = GeoApp(mcom_data, ltemdtdata)
                    self.geodata.run_geo_app()

                with con2:
                    # self.chart_generator.create_charts_tastate(ta_state)
                    st.markdown(
                        *styling(
                            f"📶 TA Remark for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h4",
                        )
                    )
                    tafinal = calculate_rf(isd_data, tastate_data)
                    st.table(tafinal)

                col1 = st.columns(1)[0]
                con1 = col1.container()
                with col1:
                    st.markdown(
                        *styling(
                            f"📶 TA State for Site {siteid}",
                            font_size=24,
                            text_align="left",
                            tag="h4",
                        )
                    )
                    self.chart_generator.create_charts_tastate(
                        ta_state, "eutrancell_new"
                    )

                with col1:
                    st.markdown(
                        *styling(
                            f"⛐ Drive Test Verification for Site {siteid}",
                            font_size=28,
                            text_align="left",
                            tag="h6",
                        )
                    )

                col1, col2, col3 = st.columns([1, 1, 1])
                con1 = col1.container(border=True)
                con2 = col2.container(border=True)
                con3 = col3.container(border=True)

                with con1:
                    with stylable_container(
                        key="rsrpsinr",
                        css_styles="""
                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            max-width: 100%;
                            position: relative;
                            top: 0px;
                        }
                        .custom-container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px);
                        }
                        """,
                    ):
                        st.markdown(
                            *styling(
                                f"RSRP & SINR {siteid}",
                                font_size=24,
                                text_align="center",
                                tag="h6",
                            )
                        )
                        if os.path.exists(folder):
                            idlersrp = os.path.join(folder, "idlersrp.png")
                            sinr = os.path.join(folder, "sinr.png")
                            if os.path.exists(idlersrp) and os.path.exists(sinr):
                                st.markdown(*styling("RSRP IDLE", font_size=15))
                                st.image(idlersrp, caption=None, use_column_width=True)
                                st.markdown(*styling("SINR", font_size=15))
                                st.image(sinr, caption=None, use_column_width=True)
                            else:
                                st.error(
                                    f"Please upload the images: {idlersrp} & {sinr}"
                                )
                        else:
                            st.error(f"Path does not exist: {folder}")

                with con2:
                    with stylable_container(
                        key="pci",
                        css_styles="""
                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            max-width: 100%;
                            position: relative;
                            top: 0px;
                        }
                        .custom-container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px);
                        }
                        """,
                    ):
                        st.markdown(
                            *styling(
                                f"PCI {siteid}",
                                font_size=24,
                                text_align="center",
                                tag="h6",
                            )
                        )
                        if os.path.exists(folder):
                            pci = os.path.join(folder, "pci.png")
                            if os.path.exists(pci):
                                st.markdown(*styling("PCI", font_size=15))
                                st.image(pci, caption=None, use_column_width=True)
                            else:
                                st.error(f"Please upload the images: {pci}")
                        else:
                            st.error(f"Path does not exist: {folder}")

                with con3:
                    with stylable_container(
                        key="pci",
                        css_styles="""
                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            max-width: 100%;
                            position: relative;
                            top: 0px;
                        }
                        .custom-container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px);
                        }
                        """,
                    ):
                        st.markdown(
                            *styling(
                                f"THROUGHPUT {siteid}",
                                font_size=24,
                                text_align="center",
                                tag="h6",
                            )
                        )
                        if os.path.exists(folder):
                            dlthp = os.path.join(folder, "dlthp.png")
                            ulthp = os.path.join(folder, "ulthp.png")
                            if os.path.exists(dlthp) & os.path.exists(ulthp):
                                st.markdown(
                                    *styling("THROUGHPUT DOWNLOAD", font_size=15)
                                )
                                st.image(dlthp, caption=None, use_column_width=True)
                                st.markdown(*styling("THROUGHPUT UPLOAD", font_size=15))
                                st.image(ulthp, caption=None, use_column_width=True)
                            else:
                                st.error(f"Please upload the images: {dlthp} & {ulthp}")
                        else:
                            st.error(f"Path does not exist: {folder}")

                (
                    col1,
                    _,
                    _,
                ) = st.columns([1, 1, 1])
                with col1:
                    st.markdown(
                        *styling(
                            f"⛐ CET Static Verification for Site {siteid}",
                            font_size=28,
                            text_align="left",
                            tag="h6",
                        )
                    )

                col1, col2, col3 = st.columns([1, 1, 1], gap="small")
                con1 = col1.container(border=True)
                con2 = col2.container(border=True)
                con3 = col3.container(border=True)

                with con1:
                    with stylable_container(
                        key="sec1",
                        css_styles="""
                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            max-width: 100%;
                            position: relative;
                            top: 0px;
                        }
                        .custom-container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px);
                        }
                        """,
                    ):
                        st.markdown(
                            *styling(
                                f"SECTOR 1 {siteid}",
                                font_size=24,
                                text_align="center",
                                tag="h6",
                            )
                        )
                        if os.path.exists(folder):
                            gridsec1 = os.path.join(folder, "gridsec1.jpg")
                            speedtestsec1 = os.path.join(folder, "speedtestsec1.jpg")
                            servicemodesec1 = os.path.join(
                                folder, "servicemodesec1.jpg"
                            )
                            gridservicemodesec1 = os.path.join(
                                folder, "gridservicemodesec1.jpg"
                            )

                            existing_images = [
                                img
                                for img in [
                                    gridsec1,
                                    speedtestsec1,
                                    gridservicemodesec1,
                                    servicemodesec1,
                                ]
                                if os.path.exists(img)
                            ]

                            if existing_images:
                                for img in existing_images:
                                    st.image(img, caption=None, use_column_width=True)
                            else:
                                st.error(
                                    "No Functionality Test for Sector 1. (gridsec1, speedtestsec1, gridservicemodesec1, servicemodesec1)"
                                )
                        else:
                            st.error(f"Path does not exist: {folder}")

                with con2:
                    with stylable_container(
                        key="sec2",
                        css_styles="""
                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            max-width: 100%;
                            position: relative;
                            top: 0px;
                        }
                        .custom-container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px);
                        }
                        """,
                    ):
                        st.markdown(
                            *styling(
                                f"SECTOR 2 {siteid}",
                                font_size=24,
                                text_align="center",
                                tag="h6",
                            )
                        )
                        if os.path.exists(folder):
                            gridsec2 = os.path.join(folder, "gridsec2.jpg")
                            speedtestsec2 = os.path.join(folder, "speedtestsec2.jpg")
                            servicemodesec2 = os.path.join(
                                folder, "servicemodesec2.jpg"
                            )
                            gridservicemodesec2 = os.path.join(
                                folder, "gridservicemodesec2.jpg"
                            )

                            existing_images = [
                                img
                                for img in [
                                    gridsec2,
                                    speedtestsec2,
                                    gridservicemodesec2,
                                    servicemodesec2,
                                ]
                                if os.path.exists(img)
                            ]

                            if existing_images:
                                for img in existing_images:
                                    st.image(img, caption=None, use_column_width=True)
                            else:
                                st.error(
                                    "No Functionality Test for Sector 2. (gridsec2, speedtestsec2, gridservicemodesec2, servicemodesec2)"
                                )
                        else:
                            st.error(f"Path does not exist: {folder}")

                with con3:
                    with stylable_container(
                        key="sec3",
                        css_styles="""
                        img {
                            display: block;
                            margin-left: auto;
                            margin-right: auto;
                            width: 100%;
                            max-width: 100%;
                            position: relative;
                            top: 0px;
                        }
                        .custom-container {
                            background-color: #F5F5F5;
                            border: 2px solid rgba(49, 51, 63, 0.2);
                            border-radius: 0.5rem;
                            padding: calc(1em - 1px);
                        }
                        """,
                    ):
                        st.markdown(
                            *styling(
                                f"SECTOR 3 {siteid}",
                                font_size=24,
                                text_align="center",
                                tag="h6",
                            )
                        )

                        if os.path.exists(folder):
                            gridsec3 = os.path.join(folder, "gridsec3.jpg")
                            speedtestsec3 = os.path.join(folder, "speedtestsec3.jpg")
                            servicemodesec3 = os.path.join(
                                folder, "servicemodesec3.jpg"
                            )
                            gridservicemodesec3 = os.path.join(
                                folder, "gridservicemodesec3.jpg"
                            )

                            existing_images = [
                                img
                                for img in [
                                    gridsec3,
                                    speedtestsec3,
                                    gridservicemodesec3,
                                    servicemodesec3,
                                ]
                                if os.path.exists(img)
                            ]

                            if existing_images:
                                for img in existing_images:
                                    st.image(img, caption=None, use_column_width=True)
                            else:
                                st.error(
                                    "No Functionality Test for Sector 3. (gridsec3, speedtestsec3, gridservicemodesec3, servicemodesec3)"
                                )
                        else:
                            st.error(f"Path does not exist: {folder}")

                session.close()
        else:
            if "mcom_data" in st.session_state and "ltemdtdata" in st.session_state:
                self.geodata = GeoApp(
                    st.session_state.mcom_data, st.session_state.ltemdtdata
                )
                self.geodata.run_geo_app()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.markdown(
        """
            <style>
            [data-testid="collapsedControl"] {
                    display: none;
                }
            #MainMenu, header, footer {visibility: hidden;}
            .appview-container .main .block-container {
                padding-top: 1px;
                padding-left: 1rem;
                padding-right: 1rem;
                padding-bottom: 1rem;
            }
            </style>
            """,
        unsafe_allow_html=True,
    )
    app = App()
    app.run()
