@echo off
REM Build script for SplitFace applications on Windows
REM Creates standalone executables for S1, S2, and S3

setlocal enabledelayedexpansion

echo ==========================================
echo SplitFace v2.0.0 - Windows Build Script
echo ==========================================
echo.

REM Check if PyInstaller is installed
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: PyInstaller not found. Please install it:
    echo   pip install pyinstaller
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.

REM Function to build an application
call :build_app "Face Mirror" "S1 Face Mirror" "Face_Mirror.spec"
call :build_app "Action Coder" "S2 Action Coder" "Action_Coder.spec"
call :build_app "Data Analysis" "S3 Data Analysis" "Data_Analysis.spec"

echo ==========================================
echo Build Complete!
echo ==========================================
echo.
echo Applications built:
echo   * S1 Face Mirror\dist\Face Mirror\
echo   * S2 Action Coder\dist\Action Coder\
echo   * S3 Data Analysis\dist\Data Analysis\
echo.
echo To run the applications, navigate to the dist folders and run:
echo   "Face Mirror.exe"
echo   "Action Coder.exe"
echo   "Data Analysis.exe"
echo.
goto :end

:build_app
set APP_NAME=%~1
set APP_DIR=%~2
set SPEC_FILE=%~3

echo ==========================================
echo Building: %APP_NAME%
echo ==========================================

cd "%APP_DIR%"

REM Clean previous builds
if exist "build" (
    echo Cleaning previous build artifacts...
    rmdir /s /q build
)

if exist "dist" (
    echo Cleaning previous distribution...
    rmdir /s /q dist
)

REM Run PyInstaller
echo Running PyInstaller...
pyinstaller "%SPEC_FILE%"

if errorlevel 1 (
    echo X %APP_NAME% build failed!
    cd ..
    exit /b 1
)

echo √ %APP_NAME% built successfully!
echo   Location: %APP_DIR%\dist\%APP_NAME%\
echo.

cd ..
goto :eof

:end
endlocal
