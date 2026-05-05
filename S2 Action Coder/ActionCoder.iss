; Inno Setup script for S2 Action Coder (Windows)
;
; Compile via:
;     iscc /DAppVersion=1.1.1 /OC:\path\to\output ActionCoder.iss
;
; Or use ..\build_s2_windows.ps1 which wires this together with PyInstaller.

#ifndef AppVersion
#define AppVersion "1.1.1"
#endif

#define AppName       "S2 Action Coder"
#define AppPublisher  "SplitFace"
#define AppURL        "https://github.com/johnwilsoniv/FaceMirror"
#define AppExeName    "S2 Action Coder.exe"
; PyInstaller --onedir output (relative to this .iss file's directory)
#define AppSourceDir  "dist\S2 Action Coder"

[Setup]
; A unique GUID for the application -- distinct from S1.
AppId={{E9D8B7C3-1A2F-4C9D-9A4F-ACTIONCODER001}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases

DefaultDirName={autopf}\FaceMirror\{#AppName}
DefaultGroupName=FaceMirror
DisableProgramGroupPage=yes
DisableDirPage=no
AllowNoIcons=yes

OutputBaseFilename=ActionCoder-S2-{#AppVersion}-win64
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern

; --- Disk spanning for GH free-tier 2 GB asset limit -----------------------
; S2's bundle is ~2 GB (smaller than S1 because no CUDA), so this MAY result
; in a single .exe. DiskSpanning is set conservatively in case future
; deps push the bundle over 2 GB.
DiskSpanning=yes
DiskSliceSize=1700000000
DiskClusterSize=4096
SlicesPerDisk=1

ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

LicenseFile=..\LICENSE
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}

PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; LAV Filters provides the DirectShow HEVC/H.265 decoder that QtMultimedia uses
; to play iOS .MOV files. Without it, S2's Qt video pipeline returns
; InvalidMedia and the play button silently no-ops on iOS recordings.
; Bundled redistributable: LAVFilters-0.81-Installer.exe (GPLv2, ~16 MB,
; downloaded from https://github.com/Nevcairiel/LAVFilters/releases/tag/0.81 ,
; SHA256 c00ea85173739871d2957aad3b5e0a413cc7d070d5993d3b1eb150ae91b183b3).
; The installer itself is built with Inno Setup so the silent flags work
; cleanly. We tag it with onlyifdoesntexist on a marker registry key so
; reinstalls of S2 don't reinstall LAV every time.
Source: "installer_redist\LAVFilters-0.81-Installer.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
; Install LAV Filters silently if not already present. The Check function
; pokes at HKLM\Software\LAV Filters to decide. /VERYSILENT suppresses the
; UI entirely; /NORESTART avoids the post-install reboot prompt; /SP- skips
; the disk-space confirmation page.
Filename: "{tmp}\LAVFilters-0.81-Installer.exe"; \
    Parameters: "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP-"; \
    StatusMsg: "Installing LAV Filters (HEVC video codec)..."; \
    Flags: waituntilterminated; \
    Check: not LAVFiltersAlreadyInstalled
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function LAVFiltersAlreadyInstalled: Boolean;
var
    Version: String;
begin
    // Both 32- and 64-bit installs land under HKLM\Software\LAV Filters by
    // default. We only need *some* version installed -- LAV is generally
    // backwards-compatible, and reinstalling on top of a newer version
    // would be obnoxious. Future-proof: if S2 ships a newer LAV in the
    // future and wants to force-upgrade, compare Version against a minimum.
    Result := RegQueryStringValue(HKLM, 'Software\LAV Filters', 'Version', Version)
           or RegQueryStringValue(HKLM64, 'Software\LAV Filters', 'Version', Version);
end;
