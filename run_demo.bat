@echo off
rem Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Consider creating one with 'python -m venv venv'.
)
python demo.py
pause
