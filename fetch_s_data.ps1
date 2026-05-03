#Requires -Version 5.1
<#
.SYNOPSIS
    Pull the S Data cohort videos from a Mac on the same LAN over SMB into
    a local Windows folder, ready for batch processing through S1 Face Mirror.

.DESCRIPTION
    The full S Data corpus lives at:
        \\<mac-ip>\johnwilsoniv\Documents\SplitFace Open3\S Data\
    organized into Lido Controls / Normal Cohort / Paralysis Cohort cohort
    subfolders containing the source videos. This script robocopies just the
    cohort video files (skipping training_data.h5 and training_data_dir which
    are model-training artifacts not needed for S1 reprocessing) into:
        $env:USERPROFILE\Documents\SplitFace_Open3_input\

    Skips files already present (resume-friendly), 8 threads, sane retry
    knobs for transient SMB hiccups.

.PARAMETER SourceShare
    UNC root of the Mac SMB share (e.g. "\\192.168.1.33\johnwilsoniv").

.PARAMETER DestBase
    Local Windows folder. Default: ~\Documents\SplitFace_Open3_input.

.EXAMPLE
    .\fetch_s_data.ps1 -SourceShare "\\192.168.1.33\johnwilsoniv"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$SourceShare,
    [string]$DestBase = "$env:USERPROFILE\Documents\SplitFace_Open3_input"
)

$ErrorActionPreference = 'Stop'
$src = Join-Path $SourceShare 'Documents\SplitFace Open3\S Data'
if (-not (Test-Path $src)) {
    throw "Source path not accessible: $src -- did you authenticate to the share via Explorer first?"
}

$cohorts = @('Lido Controls', 'Normal Cohort', 'Paralysis Cohort', 'Lido Affected')

if (-not (Test-Path $DestBase)) { New-Item -ItemType Directory -Path $DestBase | Out-Null }

$totalFiles = 0
$totalBytes = 0
foreach ($coh in $cohorts) {
    $srcCoh = Join-Path $src $coh
    if (-not (Test-Path $srcCoh)) {
        Write-Warning "Cohort missing on Mac: $srcCoh -- skipping"
        continue
    }
    $dstCoh = Join-Path $DestBase $coh
    Write-Host ("=" * 60)
    Write-Host "Cohort: $coh"
    Write-Host ("=" * 60)

    # /E recurses into per-patient subdirs (if any), /MT:8 = 8 threads,
    # /R:2 /W:5 = limit retries, /XF skips the model-training HDF5 if it
    # were under here, /XD skips training_data_dir.
    & robocopy.exe $srcCoh $dstCoh /E /MT:8 /R:2 /W:5 /NDL /NP /NJH /NJS /TEE `
        /XF '*.h5' /XD 'training_data_dir' '__pycache__'
    $rc = $LASTEXITCODE
    if ($rc -ge 8) { throw "robocopy failed copying $coh (exit $rc)" }

    $files = Get-ChildItem $dstCoh -Recurse -File -ErrorAction SilentlyContinue
    $bytes = ($files | Measure-Object Length -Sum).Sum
    Write-Host ("  -> {0} files, {1:N1} MB local" -f $files.Count, ($bytes/1MB))
    $totalFiles += $files.Count
    $totalBytes += $bytes
}

Write-Host ""
Write-Host ("=" * 60)
Write-Host ("DONE -- {0} files, {1:N2} GB at {2}" -f $totalFiles, ($totalBytes/1GB), $DestBase)
Write-Host ("=" * 60)

# Also list the .mp4 video files specifically (what S1 actually processes)
$videos = Get-ChildItem $DestBase -Recurse -File -Include *.mp4,*.MOV,*.mov,*.mkv -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "Video files ready for S1 batch:" -ForegroundColor Green
Write-Host ("  count = {0}, total = {1:N1} GB" -f $videos.Count, (($videos | Measure-Object Length -Sum).Sum / 1GB))
