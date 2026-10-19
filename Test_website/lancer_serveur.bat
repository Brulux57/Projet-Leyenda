@echo off
chcp 65001 >nul
cd /d "%~dp0"
title La Ruche - serveur local
echo === La Ruche : serveur local du site de test ===
echo.
rem Le serveur tourne dans WSL (TensorFlow sur la carte graphique, comme les notebooks). Sans WSL, Python de Windows.
rem Une fois le serveur pret, le site souvre tout seul dans le navigateur : http://localhost:8765/
rem Pour arreter le serveur : fermer cette fenetre (ou Ctrl+C).
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 http://localhost:8765/api/etat | Out-Null; Start-Process http://localhost:8765/; break}catch{Start-Sleep -Milliseconds 500}}"
where wsl >nul 2>nul
if not errorlevel 1 (
  wsl.exe --cd "%~dp0." -- bash ./lancer_serveur.sh %*
  goto fin
)
set "PY="
if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 (set "PY=python") else (set "PY=py")
)
"%PY%" serveur.py %*
:fin
echo.
pause
