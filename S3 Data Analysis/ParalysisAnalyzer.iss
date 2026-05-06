; Inno Setup script for S3 Data Analysis (Windows)
;
; Compile via:
;     iscc /DAppVersion=1.1.1 /OC:\path\to\output ParalysisAnalyzer.iss
;
; Or use ..\build_s3_windows.ps1 which wires this together with PyInstaller.

#ifndef AppVersion
#define AppVersion "1.1.1"
#endif

#define AppName       "S3 Data Analysis"
#define AppPublisher  "SplitFace"
#define AppURL        "https://github.com/johnwilsoniv/FaceMirror"
#define AppExeName    "S3 Data Analysis.exe"
; PyInstaller --onedir output (relative to this .iss file's directory)
#define AppSourceDir  "dist\S3 Data Analysis"

[Setup]
; A unique GUID for the application -- distinct from S1 + S2.
AppId={{F7C3D9A1-5B8E-4F2D-B1A4-PARALYSIS3001}
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

OutputBaseFilename=ParalysisAnalyzer-S3-{#AppVersion}-win64
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern

; --- Disk spanning for GH free-tier 2 GB asset limit -----------------------
; S3's bundle is small (~1 GB -- pure pandas + sklearn + xgboost, no torch /
; whisper / DirectShow nightmare) so this should produce a single .exe with
; one .bin slice. DiskSpanning is set conservatively in case future deps
; push the bundle over 2 GB.
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
