@echo off
setlocal
rem Builds ATV Offroad Fury 2 for PC from your own disc image:
rem   setup.bat "D:\Games\ATV Offroad Fury 2 (USA) (v2.00).iso"
rem or drag the ISO onto this file. More options: setup.bat --help
rem Needs Visual Studio 2026 ("Desktop development with C++"), Python 3.8+ and git.
set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 8))" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 8))" >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [ERROR] Python 3.8 or newer was not found. Install it from https://www.python.org/downloads/
    echo         ^(tick "Add python.exe to PATH" in the installer^), then run this again.
    set "RC=1"
    goto end
)
set "ISO=%~1"
if not defined ISO (
    set /p "ISO=Path of your ATV Offroad Fury 2 (USA) (v2.00) ISO: "
)
if not defined ISO (
    echo [ERROR] No ISO given.
    set "RC=1"
    goto end
)
if "%~1"=="" (
    %PY% -I "%~dp0tools\setup\build_atv2.py" "%ISO:"=%"
) else (
    %PY% -I "%~dp0tools\setup\build_atv2.py" %*
)
set "RC=%ERRORLEVEL%"
:end
if not defined ATV2_SETUP_NO_PAUSE pause
exit /b %RC%
