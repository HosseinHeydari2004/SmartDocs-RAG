"""Guards against unreadable colours: every explicit colour pair must meet WCAG AA, and no rule may
set a text colour without its own background (it would vanish in the opposite Streamlit theme)."""
import re
import sys
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parent.parent / "ui"
sys.path.insert(0, str(UI))
import common  # noqa: E402


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    chans = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in chans]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("name,pair", list(common.PILL_COLORS.items()) + list(common.BUTTON_COLORS.items()))
def test_colour_pairs_meet_wcag_aa(name, pair):
    assert contrast(*pair) >= 4.5, f"{name}: {contrast(*pair):.2f}:1"


def test_no_text_colour_without_own_background():
    css = common.CSS
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if re.search(r"(?<![-\w])color\s*:", body):
            assert "background" in body, f"rule '{selector.strip()}' sets color without background"


def test_no_hardcoded_page_or_text_theme_colours():
    assert "#0f172a" not in common.CSS and "#475569" not in common.CSS  # old fixed dark text colours
    config = (UI.parent / ".streamlit" / "config.toml").read_text()
    for key in ("backgroundColor", "textColor", "secondaryBackgroundColor"):
        assert key not in config  # let Streamlit's light/dark theme decide
