; Inno Setup script for S2 Action Coder (Windows)
;
; Compile via:
;     iscc /DAppVersion=1.0.0 /OC:\path\to\output ActionCoder.iss
;
; Or use ..\build_s2_windows.ps1 which wires this together with PyInstaller.

#ifndef AppVersion
#define AppVersion "1.0.0"
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

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
