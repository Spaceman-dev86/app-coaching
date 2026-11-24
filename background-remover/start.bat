@echo off
cd /d C:\workspace\background-remover
echo Stopping any old server on port 7860...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1
call .venv\Scripts\activate.bat
echo Starting Background Remover (MP4 -^> WebP animé)...
python main.py
pause
