import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from color_palette import ColorPalette
from constants import CSS_CONTAINER_STYLE


class ChartGenerator:
    def __init__(self):
        self.palette = ColorPalette()

    def get_colors(self, num_colors: int) -> list[str]:
        return [self.palette.get_color(i) for i in range(num_colors)]

    def create_line_chart(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        group: str,
        title: str,
        xrule: bool = False,
    ):
        unique_groups = df[group].nunique()
        colors = self.get_colors(unique_groups)
        fig = px.line(
            df, x=x, y=y, color=group, title=title, color_discrete_sequence=colors
        )
        if xrule and "xrule" in st.session_state:
            fig.add_vline(
                x=st.session_state["xrule"], line_dash="dash", line_color="#808080"
            )
        fig.update_layout(
            paper_bgcolor="#F5F5F5",
            plot_bgcolor="#F5F5F5",
            height=350,
            legend=dict(orientation="h", y=-0.2),
        )
        with stylable_container("chart", CSS_CONTAINER_STYLE):
            st.plotly_chart(fig, use_container_width=True)

    def create_vswr_chart(self, df: pd.DataFrame, x1: str, x2: str, y: str, title: str):
        avg_df = df.groupby([x1, x2])[y].mean().reset_index()
        unique_rrus = avg_df[x2].nunique()
        colors = self.get_colors(unique_rrus)
        fig = go.Figure()
        for idx, rru in enumerate(avg_df[x2].unique()):
            data = avg_df[avg_df[x2] == rru]
            fig.add_trace(
                go.Bar(
                    x=data[x1],
                    y=data[y],
                    name=rru,
                    marker_color=colors[idx % len(colors)],
                )
            )
        fig.add_hline(y=1.3, line_dash="dashdot", line_color="#F70000")
        fig.update_layout(
            barmode="group",
            title=title,
            paper_bgcolor="#F5F5F5",
            plot_bgcolor="#F5F5F5",
            height=350,
        )
        with stylable_container("vswr", CSS_CONTAINER_STYLE):
            st.plotly_chart(fig, use_container_width=True)
