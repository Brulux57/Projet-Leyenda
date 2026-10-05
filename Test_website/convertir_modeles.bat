@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === La Ruche : conversion des modeles ===
echo.
where python >nul 2>nul
if %errorlevel%==0 (python convertir_modeles.py %*) else (py convertir_modeles.py %*)
echo.
pause
