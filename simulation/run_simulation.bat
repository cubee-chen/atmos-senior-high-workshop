@echo off
echo Starting Taipei Thunderstorm Simulation...
echo.
python -m pip install -r requirements.txt
echo.
echo ----------------------------------------------------
echo IMPORTANT: Please ensure you have edited the parameters.txt file
echo with your desired simulation inputs before proceeding.
echo ----------------------------------------------------
echo.
pause
python main.py
echo.
pause