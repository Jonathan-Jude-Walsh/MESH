import customtkinter as ctk

# ============================================================
# MESH Theme
# ============================================================

APP_TITLE = "MESH"

WINDOW_WIDTH = 1600

WINDOW_HEIGHT = 900

SIDEBAR_WIDTH = 250

# ============================================================
# Colors
# ============================================================

ACCENT = "#4A90E2"

SUCCESS = "#28A745"

WARNING = "#FFC107"

ERROR = "#DC3545"

# ============================================================
# Theme Setup
# ============================================================

def setup_theme():

    ctk.set_appearance_mode(
        "dark"
    )

    ctk.set_default_color_theme(
        "blue"
    )