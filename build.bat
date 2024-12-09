@echo off

set "VENV_DIR=venv"
set "BUILD_SYSTEM_DIR=build_system"
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
    .\%VENV_DIR%\Scripts\python %BUILD_SYSTEM_DIR%\main.py %* || exit
    call deactivate || exit
)

if "%CLEAN%"=="true" if exist "%VENV_DIR%" (
    echo Removing virtual Python environment...
    rd /s /q "%VENV_DIR%" || exit

    if exist "%BUILD_SYSTEM_DIR%\build" (
        rd /s /q "%BUILD_SYSTEM_DIR%\build" || exit
    )
)
