from typing import Literal


def styling(
    text: str,
    tag: Literal["h1", "h2", "h3", "h4", "h5", "h6", "p"] = "h2",
    text_align: Literal["left", "right", "center", "justify"] = "center",
    font_size: int = 32,
    font_family: str = "Plus Jakarta Sans SemiBold",
    background_color: str = "transparent",
    font_color: str = "black",
) -> tuple[str, bool]:
    style = f"text-align: {text_align}; font-size: {font_size}px; font-family: {font_family}; background-color: {background_color}; color: {font_color};"
    styled_text = f'<{tag} style="{style}">{text}</{tag}>'
    return styled_text, True
