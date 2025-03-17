import os

# File paths
CONFIG_PATH = ".streamlit/secrets.toml"
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)

# CSS styles
CSS_CONTAINER_STYLE = """
    {
        background-color: #F5F5F5;
        border: 2px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: calc(1em - 1px)
    }
"""
