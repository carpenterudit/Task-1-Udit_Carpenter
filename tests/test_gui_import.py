import pytest

def test_gui_import():
 pytest.importorskip('streamlit')
 import app
 assert app is not None
