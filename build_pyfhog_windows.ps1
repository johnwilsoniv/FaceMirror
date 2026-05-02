#Requires -Version 5.1
<#
.SYNOPSIS
    Build and install the Windows-patched pyfhog v0.1.4 into the active venv.

.DESCRIPTION
    Why this script exists:
      pyfhog v0.1.3 (the latest version with a PyPI cp310-win_amd64 wheel)
      has a TRANSPOSED HOG INDEXING BUG in src/cpp/fhog_wrapper.cpp:

          ptr[idx++] = hog[x][y](o);   // 0.1.3 -- WRONG, doesn't match OpenFace 2.2

      Running pyfaceau on Windows with v0.1.3 produces HOG features in the
      wrong layout, which feed the AU SVR predictors and produce AU output
      that is uncorrelated (or NEGATIVELY correlated) with the macOS goldens
      and the C++ ground truth. Median Pearson r ~ 0.47, with several AUs
      below zero.

      pyfhog v0.1.4 fixes this:

          ptr[idx++] = hog[y][x](o);   // 0.1.4 -- matches OpenFace Face_utils.cpp

      Same release also re-enabled dlib's test_for_odr_violations.h sentinels
      that 0.1.3 had commented out as a Windows workaround. Those sentinels
      do not link on Windows MSVC because pyfhog uses dlib in header-only
      mode and never compiles dlib/all/source.cpp. So PyPI 0.1.4 is sdist-only
      and fails to build on Windows with:

          fhog_wrapper.obj : error LNK2001: unresolved external symbol
              DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_19_13_0
          fhog_wrapper.obj : error LNK2001: unresolved external symbol
              USER_ERROR__inconsistent_build_configuration__see_dlib_faq_1_

    This script clones pyfhog at v0.1.4, replaces only that one header file
    with the vendored Windows-friendly version (matching the 0.1.3 state of
    the same file), and pip-installs from the patched source. The HOG kernel
    code (the part we actually need fixed) is untouched.

    After running this script, `pip show pyfhog` reports 0.1.4, and pyfaceau
    AU output on Windows correlates with macOS goldens at median r > 0.98.

.PARAMETER Venv
    Path to the venv to install into. Defaults to S1_FaceMirror/.venv.

.PARAMETER PyfhogTag
    Git tag/SHA of pyfhog to clone. Defaults to v0.1.4.

.EXAMPLE
    .\build_pyfhog_windows.ps1
#>
[CmdletBinding()]
param(
    [string]$Venv = (Join-Path $PSScriptRoot "S1_FaceMirror\.venv"),
    [string]$PyfhogTag = 'v0.1.4'
)

$ErrorActionPreference = 'Stop'

$python = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "Venv python not found at $python -- create the venv and pip install -r requirements-windows-cuda.txt first."
}

# vswhere/MSVC sanity check
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "MSVC Build Tools 2022 not detected -- install via winget Microsoft.VisualStudio.2022.BuildTools first."
}

$buildDir = Join-Path $env:TEMP "pyfhog-build-$(Get-Random)"
$patchSource = Join-Path $PSScriptRoot 'S1_FaceMirror\pyfhog_windows_patch\test_for_odr_violations.h'
if (-not (Test-Path $patchSource)) {
    throw "Vendored patch missing at $patchSource"
}

Write-Host "Building patched pyfhog $PyfhogTag in $buildDir"
try {
    git clone --depth 1 --branch $PyfhogTag https://github.com/johnwilsoniv/pyfhog.git $buildDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }

    $patchTarget = Join-Path $buildDir 'src\cpp\dlib\test_for_odr_violations.h'
    Write-Host "Applying patch: $patchSource -> $patchTarget"
    Copy-Item -Force -Path $patchSource -Destination $patchTarget

    Write-Host "Building wheel + installing into $Venv"
    Push-Location $buildDir
    try {
        & $python -m pip install . --no-deps --force-reinstall
        if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }

    Write-Host ""
    & $python -m pip show pyfhog | Select-String -Pattern '^(Name|Version|Location):'
    Write-Host "pyfhog Windows-patched build complete." -ForegroundColor Green
} finally {
    if (Test-Path $buildDir) {
        Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
    }
}
