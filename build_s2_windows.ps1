#Requires -Version 5.1
<#
.SYNOPSIS
    Build the Windows installer for S2 Action Coder.

.DESCRIPTION
    Mirrors build_windows.ps1 (which targets S1) for S2:
        1. PyInstaller (--onedir) -> dist\S2 Action Coder\
        2. Inno Setup ISCC -> single-file (or disk-spanned) .exe installer

    S2's existing PyInstaller spec (Action_Coder.spec) is already
    cross-platform; this script just supplies the Windows build env
    (active venv + ISCC.exe) and wires the steps together.

    Unlike S1, S2 has no CUDA dependency -- it ships CPU faster-whisper
    and does Whisper transcription on CPU. That keeps the bundle ~2 GB
    instead of S1's 6 GB. We still use Inno Setup DiskSpanning for GH
    free-tier consistency.

.PARAMETER Version
    Installer version string. Default: 1.0.0.

.PARAMETER SkipBuild
    Skip the PyInstaller step (useful when iterating on the .iss file).
#>
[CmdletBinding()]
param(
    [string]$Version = '1.0.0',
    [switch]$SkipBuild,
    [string]$InnoSetupPath = ''
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$S2Dir = Join-Path $ScriptDir 'S2 Action Coder'
$DistDir = Join-Path $S2Dir 'dist'
$BuildDir = Join-Path $S2Dir 'build'
$InstallerOutDir = Join-Path $ScriptDir 'installer_output_s2'
$IssFile = Join-Path $S2Dir 'ActionCoder.iss'

function Write-Section([string]$Message) {
    Write-Host ''
    Write-Host '========================================' -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host '========================================' -ForegroundColor Cyan
}

function Resolve-InnoSetup {
    if ($InnoSetupPath -and (Test-Path $InnoSetupPath)) { return $InnoSetupPath }
    $candidates = @(
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($p in $candidates) { if ($p -and (Test-Path $p)) { return $p } }
    return $null
}

# ----- Sanity checks -----
Write-Section "Build S2 Action Coder v$Version (Windows)"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'python not found on PATH. Activate S2"s venv first: cd "S2 Action Coder"; .\.venv\Scripts\Activate.ps1'
}

if (-not $SkipBuild) {
    if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
        throw 'pyinstaller not found. Run: pip install -r "S2 Action Coder\requirements-windows.txt"'
    }
}

$Iscc = Resolve-InnoSetup
if (-not $Iscc) {
    Write-Warning 'Inno Setup 6 not found -- skipping installer compile. Install from https://jrsoftware.org/isdl.php to produce a .exe installer.'
}
if ($Iscc) { Write-Host "Using ISCC.exe at $Iscc" }

# ----- PyInstaller -----
if (-not $SkipBuild) {
    Write-Section 'Step 1: PyInstaller --onedir'
    Push-Location $S2Dir
    try {
        if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
        if (Test-Path $DistDir)  { Remove-Item -Recurse -Force $DistDir }
        # Same cmd.exe wrapper trick as build_windows.ps1: PowerShell 5.1
        # under ErrorActionPreference='Stop' raises NativeCommandError on
        # PyInstaller's INFO-on-stderr output. cmd.exe sidesteps it.
        cmd /c "pyinstaller --clean --noconfirm Action_Coder.spec 2>&1"
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} else {
    Write-Host '(skipping PyInstaller per -SkipBuild)'
}

$AppDir = Join-Path $DistDir 'S2 Action Coder'
if (-not (Test-Path $AppDir)) {
    throw "Expected PyInstaller output at $AppDir but it does not exist."
}

# ----- Inno Setup (skipped if ISCC not installed) -----
if ($Iscc -and (Test-Path $IssFile)) {
    Write-Section 'Step 2: Inno Setup compile'
    if (-not (Test-Path $InstallerOutDir)) {
        New-Item -ItemType Directory -Path $InstallerOutDir | Out-Null
    }
    & $Iscc "/DAppVersion=$Version" "/O$InstallerOutDir" $IssFile
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed with exit code $LASTEXITCODE" }
} elseif (-not (Test-Path $IssFile)) {
    Write-Warning "No $IssFile -- producing only the PyInstaller dist/ folder. Zip it for portable distribution or write an .iss to make a proper installer."
}

# ----- Done -----
Write-Section 'Build complete'
Write-Host ('  PyInstaller dist: {0}' -f $AppDir) -ForegroundColor Green
if (Test-Path $InstallerOutDir) {
    Get-ChildItem $InstallerOutDir -Filter '*.exe' | ForEach-Object {
        $sizeMb = [math]::Round($_.Length / 1MB, 1)
        Write-Host ("  Installer: {0}  ({1} MB)" -f $_.FullName, $sizeMb) -ForegroundColor Green
    }
    Get-ChildItem $InstallerOutDir -Filter '*.bin' -ErrorAction SilentlyContinue | ForEach-Object {
        $sizeMb = [math]::Round($_.Length / 1MB, 1)
        Write-Host ("  Slice:     {0}  ({1} MB)" -f $_.FullName, $sizeMb) -ForegroundColor Green
    }
}
