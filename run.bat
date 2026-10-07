@echo off
cd /d %~dp0
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  pip install -r requirements-full.txt
) else (
  call .venv\Scripts\activate
)
start "" http://127.0.0.1:8000
python app.py
