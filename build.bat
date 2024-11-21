@echo off

set "VENV_DIR=venv"
set "SCONS_DIR=scons"
set "CLEAN=false"

if not "%1"=="" if "%2"=="" (
    if "%1"=="--clean" (
        set "CLEAN=true"
    )
    if "%1"=="-c" (
        set "CLEAN=true"
    )
)

if not exist "%VENV_DIR%" (
    echo Creating virtual Python environment...
    python -m venv "%VENV_DIR%" || exit
)

if exist "%VENV_DIR%" (
    call %VENV_DIR%\Scripts\activate || exit
    .\%VENV_DIR%\Scripts\python -c "import sys;sys.path.append('%SCONS_DIR%');from util.pip import Pip;Pip().install_packages('scons')" || exit
    .\%VENV_DIR%\Scripts\python -m SCons --silent --file %SCONS_DIR%\sconstruct.py %* || exit
    call deactivate || exit
)

if "%CLEAN%"=="true" if exist "%VENV_DIR%" (
    echo Removing virtual Python environment...
    rd /s /q "%VENV_DIR%" || exit

    if exist "%SCONS_DIR%\build" (
        rd /s /q "%SCONS_DIR%\build" || exit
    )
)
