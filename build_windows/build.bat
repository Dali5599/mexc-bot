@echo off
setlocal
REM ===== Build MEXC Bot Windows (one-folder) =====
if not exist venv (
  py -3 -m venv venv
)
call venv\Scripts\python -m pip install --upgrade pip
call venv\Scripts\pip install -r requirements.txt pyinstaller
REM Run PyInstaller with the spec (ensures data files are bundled)
call venv\Scripts\pyinstaller.exe build_windows\mexc_bot.spec --noconfirm --clean
echo.
echo Build finished.
echo You can run: "dist\MEXC Bot\MEXC Bot.exe"
echo.
pause
