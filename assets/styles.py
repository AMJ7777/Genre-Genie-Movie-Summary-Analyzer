"""
Styles module for the Movie Summary Analysis application.
Contains color schemes and style configurations for the GUI.
"""

# Main color scheme
PRIMARY_COLOR = "#3498db"  # Blue
SECONDARY_COLOR = "#f39c12"  # Orange
BACKGROUND_COLOR = "#f5f5f5"  # Light Gray
TEXT_COLOR = "#2c3e50"  # Dark Blue/Gray
HIGHLIGHT_COLOR = "#e74c3c"  # Red
SUCCESS_COLOR = "#2ecc71"  # Green

# Font configurations
LARGE_FONT = ("Helvetica", 14, "bold")
MEDIUM_FONT = ("Helvetica", 12)
SMALL_FONT = ("Helvetica", 10)
BUTTON_FONT = ("Helvetica", 11, "bold")

# Button styles
BUTTON_STYLE = {
    "bg": PRIMARY_COLOR,
    "fg": "white",
    "activebackground": SECONDARY_COLOR,
    "activeforeground": "white",
    "font": BUTTON_FONT,
    "relief": "flat",
    "borderwidth": 0,
    "padx": 10,
    "pady": 5
}

# Navigation button styles
NAV_BUTTON_STYLE = {
    "bg": SECONDARY_COLOR,
    "fg": "white",
    "activebackground": PRIMARY_COLOR,
    "activeforeground": "white",
    "font": BUTTON_FONT,
    "relief": "flat",
    "borderwidth": 0,
    "padx": 10,
    "pady": 5
}

# Frame styles
FRAME_STYLE = {
    "bg": BACKGROUND_COLOR,
    "padx": 10,
    "pady": 10
}

# Label styles
LABEL_STYLE = {
    "bg": BACKGROUND_COLOR,
    "fg": TEXT_COLOR,
    "font": MEDIUM_FONT
}

# Entry styles
ENTRY_STYLE = {
    "bg": "white",
    "fg": TEXT_COLOR,
    "font": MEDIUM_FONT,
    "relief": "flat",
    "borderwidth": 1
}

# List and text box styles
LISTBOX_STYLE = {
    "bg": "white",
    "fg": TEXT_COLOR,
    "font": SMALL_FONT,
    "relief": "flat",
    "borderwidth": 1,
    "selectbackground": PRIMARY_COLOR,
    "selectforeground": "white"
}

TEXT_STYLE = {
    "bg": "white",
    "fg": TEXT_COLOR,
    "font": MEDIUM_FONT,
    "relief": "flat",
    "borderwidth": 1
}

# Progress bar style
PROGRESS_STYLE = {
    "bg": BACKGROUND_COLOR,
    "fg": PRIMARY_COLOR,
    "height": 20
}
