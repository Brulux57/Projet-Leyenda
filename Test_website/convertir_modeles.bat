@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === La Ruche : conversion des modeles pour le site ===
echo.
rem Python a utiliser : environnement virtuel du projet s'il existe, sinon celui du PATH
set "PY="
if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\..\.venv\Scripts\python.exe" set "PY=%~dp0..\..\.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 (set "PY=python") else (set "PY=py")
)
"%PY%" convertir_modeles.py %*
echo.
pause
