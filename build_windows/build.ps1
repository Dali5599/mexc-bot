# Build MEXC Bot on Windows (PowerShell)
if (-not (Test-Path .\venv)) {
  py -3 -m venv venv
}
.\venv\Scripts\python -m pip install --upgrade pip
.\venv\Scripts\pip install -r requirements.txt pyinstaller
.\venv\Scripts\pyinstaller.exe build_windows\mexc_bot.spec --noconfirm --clean
Write-Host "`nBuild finished.`nRun: dist\MEXC Bot\MEXC Bot.exe`n"
