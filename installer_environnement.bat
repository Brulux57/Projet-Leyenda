@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Leyenda - installation de l'environnement Python
rem Installe dans WSL l'environnement des notebooks et du site : TensorFlow et PyTorch sur la carte graphique.
rem Le detail est dans installer_environnement.sh ; tout ce qui s'affiche est aussi ecrit dans installation.log.
where wsl >nul 2>&1
if errorlevel 1 (
  echo WSL est introuvable : installer WSL2 et Ubuntu, puis relancer ce fichier.
  goto fin
)
set "OPTIONS="
rem Ancien environnement des notebooks encore present : on propose de le supprimer (apres verification du nouveau)
wsl.exe bash -c "test -d ~/venvs/tf-gpu" >nul 2>&1
if not errorlevel 1 (
  choice /c ON /n /m "Supprimer l'ancien environnement ~/venvs/tf-gpu une fois le nouveau verifie ? [O/N] "
  if not errorlevel 2 set "OPTIONS=--supprimer-ancien"
)
wsl.exe --cd "%~dp0." -- bash ./installer_environnement.sh %OPTIONS% %*
:fin
echo.
pause
