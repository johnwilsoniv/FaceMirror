# S2 Installer Redistributables

The S2 v1.1.1+ installer (`ActionCoder.iss`) bundles **LAV Filters** so that
end users on Windows get HEVC/H.265 video decoding without having to install
a codec pack manually. The actual `LAVFilters-X.YZ-Installer.exe` binary is
**not** stored in git -- it's fetched at build time from the upstream release.

## Populate this directory before running Inno Setup

```powershell
$dest = "$PWD\LAVFilters-0.81-Installer.exe"
Invoke-WebRequest `
    -Uri "https://github.com/Nevcairiel/LAVFilters/releases/download/0.81/LAVFilters-0.81-Installer.exe" `
    -OutFile $dest
# Optional: verify SHA256
(Get-FileHash $dest -Algorithm SHA256).Hash
# Expected: c00ea85173739871d2957aad3b5e0a413cc7d070d5993d3b1eb150ae91b183b3
```

After that, `iscc /DAppVersion=1.1.1 ActionCoder.iss` (or `build_s2_windows.ps1`)
will pick up the file via the `[Files]` entry in `ActionCoder.iss`.

## License

LAV Filters is **GPLv2**. Redistributing the upstream installer as a separate
file alongside our installer is permitted; we do not modify the LAV Filters
binary, we just invoke its silent install. See
<https://github.com/Nevcairiel/LAVFilters/blob/master/LICENSE> for the full
license text.

## If you don't want HEVC support in your build

Comment out the `[Files]` line referencing `LAVFilters-*-Installer.exe` and
the `[Run]` block in `ActionCoder.iss`. The S2 installer will skip the LAV
install step entirely; users will need to install their own HEVC codec or
S2 will fail to play iOS `.MOV` source videos with the
`InvalidMedia status received from QtMultimedia` error.
