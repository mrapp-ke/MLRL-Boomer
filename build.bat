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

if not exist "%VENV_DIR%" if "%CLEAN%"=="false" (
    echo Creating virtual Python environment...
    python -m venv "%VENV_DIR%"
)

if exist "%VENV_DIR%" (
    call %VENV_DIR%\Scripts\activate ^
        && python -c "import sys;sys.path.append('%SCONS_DIR%'); import run; run.install_build_dependencies('scons')" ^
        && scons --silent --file %SCONS_DIR%\sconstruct.py %* ^
        && call deactivate
)

if "%CLEAN%"=="true" if exist "%VENV_DIR%" (
    echo Removing virtual Python environment...
    rd /s /q "%VENV_DIR%"

    if exist "%SCONS_DIR%\build" (
        rd /s /q "%SCONS_DIR%\build"
    )
)
