from typing import List, Optional

import pandas as pd
import streamlit as st
from omegaconf import DictConfig
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session as SQLAlchemySession
from sqlalchemy.orm import sessionmaker


class DatabaseSession:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.connections.postgresql
        self.engine = None

    def get_engine(self) -> create_engine | None:
        if not self.engine:
            try:
                conn_str = f"{self.cfg.dialect}://{self.cfg.username}:{self.cfg.password}@{self.cfg.host}:{self.cfg.port}/{self.cfg.database}"
                self.engine = create_engine(conn_str)
            except Exception as e:
                st.error(f"Error creating engine: {e}")
        return self.engine

    def __enter__(self) -> SQLAlchemySession:
        engine = self.get_engine()
        if engine:
            self.session = sessionmaker(bind=engine)()
            return self.session
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "session") and self.session:
            self.session.close()


class QueryManager:
    def __init__(self, engine):
        self.engine = engine

    def fetch_data(self, query: str, params: dict = None) -> pd.DataFrame:
        try:
            return pd.read_sql(text(query), self.engine, params=params)
        except Exception as e:
            st.error(f"Query error: {e}")
            return pd.DataFrame()

    def get_ltedaily_data(
        self, siteid: str, neids: list[str], start_date, end_date
    ) -> pd.DataFrame:
        base_query = 'SELECT * FROM ltedaily WHERE "SITEID" LIKE :siteid AND "DATE_ID" BETWEEN :start_date AND :end_date'
        params = {
            "siteid": f"%{siteid}%",
            "start_date": start_date,
            "end_date": end_date,
        }
        if neids:
            neid_conditions = " OR ".join(
                [f'"NEID" LIKE :neid_{i}' for i in range(len(neids))]
            )
            query = f"{base_query} AND ({neid_conditions})"
            params.update({f"neid_{i}": f"%{neid}%" for i, neid in enumerate(neids)})
        else:
            query = base_query
        return self.fetch_data(query, params)

    def get_vswr_data(self, sites: list[str], end_date) -> pd.DataFrame:
        conditions = " OR ".join(
            [f'"NE_NAME" LIKE :site_{i}' for i in range(len(sites))]
        )
        start_date = end_date - pd.Timedelta(days=3)
        query = f"""
            SELECT "DATE_ID", "NE_NAME", "RRU", "pmReturnLossAvg", "VSWR"
            FROM ltevswr
            WHERE ({conditions})
            AND "DATE_ID" BETWEEN :start_date AND :end_date
            AND "RRU" NOT LIKE '%RfPort=R%' AND "RRU" NOT LIKE '%RfPort=S%' AND "VSWR" != 0
        """
        params = {f"site_{i}": f"%{site}%" for i, site in enumerate(sites)}
        params.update({"start_date": start_date, "end_date": end_date})
        return self.fetch_data(query, params)
