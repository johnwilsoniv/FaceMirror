#Requires -Version 5.1
<#
.SYNOPSIS
    Build Windows installer for S1 Face Mirror.

.DESCRIPTION
    Parallels build_dmg.sh for the Windows + CUDA 12.8 target. Runs:
        1. PyInstaller (--onedir) to assemble dist\S1 Face Mirror\
        2. Inno Setup ISCC to compile a single-file .exe installer

.PARAMETER Version
    Installer version string. Default: 1.1.0.

.PARAMETER SkipBuild
    Skip the PyInstaller step (useful when iterating on the .iss script).

.PARAMETER InnoSetupPath
    Override the path to ISCC.exe. By default we look in the standard install
    locations under Program Files.

.EXAMPLE
    .\build_windows.ps1
    .\build_windows.ps1 -Version 1.1.0
    .\build_windows.ps1 -SkipBuild

.NOTES
    Requirements on the build machine:
        - Python 3.10 with the requirements-windows-cuda.txt env installed
        - Inno Setup 6.x (https://jrsoftware.org/isdl.php)
        - NVIDIA driver supporting CUDA 12.8
        - Optional: ffmpeg.exe + ffprobe.exe in S1_FaceMirror\bin\
#>

[CmdletBinding()]
param(
    [string]$Version = '1.1.1',
    [switch]$SkipBuild,
    [string]$InnoSetupPath = ''
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$S1Dir = Join-Path $ScriptDir 'S1_FaceMirror'
$DistDir = Join-Path $S1Dir 'dist'
$BuildDir = Join-Path $S1Dir 'build'
$InstallerOutDir = Join-Path $ScriptDir 'installer_output'
$IssFile = Join-Path $S1Dir 'FaceMirror.iss'

function Write-Section([string]$Message) {
    Write-Host ''
    Write-Host '========================================' -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host '========================================' -ForegroundColor Cyan
}

function Resolve-InnoSetup {
    if ($InnoSetupPath -and (Test-Path $InnoSetupPath)) {
        return $InnoSetupPath
    }
    $candidates = @(
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($p in $candidates) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

# ----- Sanity checks ---------------------------------------------------------
Write-Section "Build S1 Face Mirror v$Version (Windows + CUDA 12.8)"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'python not found on PATH. Install Python 3.10 and re-run.'
}

if (-not $SkipBuild) {
    if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
        throw 'pyinstaller not found on PATH. Run: pip install -r S1_FaceMirror\requirements-windows-cuda.txt'
    }
}

$Iscc = Resolve-InnoSetup
if (-not $Iscc) {
    throw 'Inno Setup 6 not found. Install from https://jrsoftware.org/isdl.php or pass -InnoSetupPath.'
}
Write-Host "Using ISCC.exe at $Iscc"

# ----- PyInstaller -----------------------------------------------------------
if (-not $SkipBuild) {
    Write-Section 'Step 1: PyInstaller --onedir'
    Push-Location $S1Dir
    try {
        if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
        if (Test-Path $DistDir)  { Remove-Item -Recurse -Force $DistDir }
        # Run pyinstaller via cmd.exe so PowerShell's strict
        # ErrorActionPreference='Stop' doesn't raise NativeCommandError on
        # PyInstaller's INFO log lines (it writes them to stderr; PS 5.1
        # treats every stderr line under Stop preference as a terminating
        # error regardless of `2>&1` redirection). cmd.exe runs pyinstaller
        # without that wrapping; we read its exit code via %ERRORLEVEL%.
        cmd /c "pyinstaller --clean --noconfirm Face_Mirror.spec 2>&1"
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} else {
    Write-Host '(skipping PyInstaller per -SkipBuild)'
}

$AppDir = Join-Path $DistDir 'S1 Face Mirror'
if (-not (Test-Path $AppDir)) {
    throw "Expected PyInstaller output at $AppDir but it does not exist."
}

# ----- Inno Setup ------------------------------------------------------------
Write-Section 'Step 2: Inno Setup compile'
if (-not (Test-Path $InstallerOutDir)) {
    New-Item -ItemType Directory -Path $InstallerOutDir | Out-Null
}

& $Iscc `
    "/DAppVersion=$Version" `
    "/O$InstallerOutDir" `
    $IssFile

if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compile failed with exit code $LASTEXITCODE"
}

# ----- Done ------------------------------------------------------------------
Write-Section 'Build complete'
Get-ChildItem $InstallerOutDir -Filter '*.exe' | ForEach-Object {
    $sizeMb = [math]::Round($_.Length / 1MB, 1)
    Write-Host ("  {0}  ({1} MB)" -f $_.FullName, $sizeMb) -ForegroundColor Green
}
