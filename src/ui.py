from typing import List, Tuple

import pandas as pd
import streamlit as st
from streamlit_extras.mandatory_date_range import date_range_picker


class StreamlitUI:
    @staticmethod
    def load_sitelist(filepath: str) -> list[list[str]]:
        with open(filepath) as f:
            return [line.strip().split(",") for line in f]

    @staticmethod
    def select_options(
        sitelist: list[list[str]], key: str, column_idx: int
    ) -> list[str]:
        options = sorted({row[column_idx] for row in sitelist})
        return st.multiselect(key, options)

    @staticmethod
    def select_date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
        default_start = pd.Timestamp.today() - pd.DateOffset(months=3)
        default_end = pd.Timestamp.today()
        return date_range_picker(
            "DATE RANGE", default_start=default_start, default_end=default_end
        )

    @staticmethod
    def select_xrule_date() -> pd.Timestamp:
        default = st.session_state.get("xrule_date", pd.Timestamp.today())
        return st.date_input("OA Date", default)
