@echo off
REM Stops the Mend app, the lab tunnel and Neo4j started by Start-Mend.bat.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Stop-Mend.ps1"
pause
