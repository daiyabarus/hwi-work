import streamlit as st
import toml
from omegaconf import DictConfig, OmegaConf

from constants import CONFIG_PATH


class ConfigLoader:
    @staticmethod
    def load() -> DictConfig:
        try:
            with open(CONFIG_PATH) as f:
                return OmegaConf.create(toml.load(f))
        except Exception as e:
            st.error(f"Error loading config: {e}")
            raise
