@echo off
REM Mend clinician assistant: starts Neo4j (no Desktop window), the LLM (lab or local) and the app. See Start-Mend.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Mend.ps1"
