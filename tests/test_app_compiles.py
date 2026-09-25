import py_compile
from pathlib import Path


def test_app_script_compiles():
    # the Streamlit script is only imported by the live walkthrough; guard syntax in the default suite
    py_compile.compile(str(Path(__file__).resolve().parent.parent / "app.py"), doraise=True)
