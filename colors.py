from typing import ClassVar


class ColorPalette:
    color_codes: ClassVar[list[str]] = [
        "#00B0F0",
        "#B60008",
        "#92D050",
        "#FFC000",
        "#D26939",
        "#9932CC",
        "#00FF00",
        "#002D80",
        "#5E2EA7",
        "#FF1493",
        "#4682B4",
        "#FF7795",
        "#32CD32",
        "#FF00FF",
        "#1E4D2B",
        "#CD853F",
        "#00BFFF",
        "#DC143C",
        "#838996",
        "#195466",
        "#40E0D0",
        "#B8860B",
        "#E9967A",
        "#8FBC8F",
        "#8B0000",
        "#00CED1",
        "#483D8B",
        "#2F4F4F",
        "#4B0150",
        "#FF8C00",
        "#556B2F",
        "#FF0000",
        "#FF6347",
        "#6B8E23",
        "#FFE4B5",
        "#008080",
        "#FFC0CB",
        "#00FF7F",
        "#FF4500",
        "#F0E68C",
        "#4169E1",
        "#F4A460",
        "#7B68EE",
        "#A0522D",
        "#C71585",
        "#66CDAA",
        "#D2691E",
        "#DB7093",
        "#DDA0DD",
        "#008000",
    ]
    last_colors: ClassVar[list[str]] = []

    @classmethod
    def get_color(cls, index: int) -> str:
        # Ensure the color list is long enough for selection
        extended_colors = cls.color_codes * ((index // len(cls.color_codes)) + 1)
        selected_color = extended_colors[index]

        # Check if the selected color is in the last 5 used colors
        while selected_color in cls.last_colors:
            index += 1
            selected_color = extended_colors[index % len(cls.color_codes)]

        # Update the last colors list
        cls.last_colors.append(selected_color)
        if len(cls.last_colors) > 1:
            cls.last_colors.pop(0)

        return selected_color
