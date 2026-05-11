#Requires -Version 5.1
<#
.SYNOPSIS
    Copy the canary corpus subdirs from a SMB-mounted Mac into a local
    Windows path so the parity test (test_tier1_windows_cuda_parity.py)
    can read videos without OpenCV's iCloud-stub timeout firing.

.DESCRIPTION
    The iCloud Drive client on this Windows box is hung in "RecallOnDataAccess"
    mode - every cloud-only file errors with "cloud operation timed out"
    inside 60s, which is shorter than OpenCV's ffmpeg read timeout. Workaround:
    pull the canary corpus directly from the Mac over SMB on the same LAN.

    Authenticate to the Mac once via File Explorer (paste \\<mac-ip> in the
    address bar) so credentials are cached. Then run this script.

.PARAMETER SourceShare
    UNC root of the Mac share that contains the SplitFace dir, e.g.
    "\\192.168.1.33\johnwilsoniv". The Documents/SplitFace path is appended.

.PARAMETER DestBase
    Local Windows folder to copy into. Defaults to Documents\SplitFace.
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$SourceShare,
    [string]$DestBase = "$env:USERPROFILE\Documents\SplitFace"
)

$ErrorActionPreference = 'Stop'

$src = Join-Path $SourceShare 'Documents\SplitFace'
if (-not (Test-Path $src)) {
    throw "Source path not accessible: $src - did you authenticate to the share via Explorer first?"
}

# Only the subdirs the parity test actually reads (~3.2 GB) - skip the 7+ GB
# of intermediate artifacts.
$subdirs = @(
    'S1O Processed Files\Face Mirror 1.0 Output',  # videos (~3 GB)
    'S2O Coded Files',                              # pyfaceau CSVs (~50 MB)
    'S2O Coded Files OF',                           # C++ CSVs (~50 MB)
    'S3O Results'                                   # combined CSVs (~50 MB)
)

if (-not (Test-Path $DestBase)) {
    New-Item -ItemType Directory -Path $DestBase | Out-Null
}

$totalBytes = 0
$totalFiles = 0
foreach ($sub in $subdirs) {
    $srcSub = Join-Path $src $sub
    $dstSub = Join-Path $DestBase $sub
    if (-not (Test-Path $srcSub)) {
        Write-Warning "Source subdir missing: $srcSub - skipping"
        continue
    }
    Write-Host ("=" * 60)
    Write-Host "Copying: $sub"
    Write-Host ("=" * 60)
    # robocopy /MIR mirrors (skips already-copied), /MT:8 = 8 threads,
    # /R:2 /W:5 = limit retries on transient SMB hiccups
    $log = Join-Path $env:TEMP "fetch_canaries_$($sub -replace '[\\:/ ]','_').log"
    & robocopy.exe $srcSub $dstSub /E /MT:8 /R:2 /W:5 /NDL /NP /NJH /NJS /TEE
    $rc = $LASTEXITCODE
    # robocopy: 0=no copy, 1=copied, 2=extras, 3=copied+extras, <8=success
    if ($rc -ge 8) { throw "robocopy failed copying $sub with exit $rc" }
    $bytes = (Get-ChildItem $dstSub -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $files = (Get-ChildItem $dstSub -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host ("  -> {0} files, {1:N1} MB" -f $files, ($bytes/1MB))
    $totalBytes += $bytes
    $totalFiles += $files
}

Write-Host ""
Write-Host ("=" * 60)
Write-Host ("DONE - {0} files, {1:N2} GB at {2}" -f $totalFiles, ($totalBytes/1GB), $DestBase)
Write-Host ("Now set: " + '$env:SPLITFACE_BASE = ' + "'$DestBase'")
Write-Host ("=" * 60)
