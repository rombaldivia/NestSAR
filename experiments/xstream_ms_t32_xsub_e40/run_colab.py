#!/usr/bin/env python3
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / "run_kaggle.py"), run_name="__main__")
