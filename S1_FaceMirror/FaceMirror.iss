; Inno Setup script for S1 Face Mirror (Windows + CUDA 12.1)
;
; Compile via:
;     iscc /DAppVersion=1.0.0 /OC:\path\to\output FaceMirror.iss
;
; Or use ..\build_windows.ps1 which wires this together with PyInstaller.

#ifndef AppVersion
#define AppVersion "1.0.0"
#endif

#define AppName       "S1 Face Mirror"
#define AppPublisher  "SplitFace"
#define AppURL        "https://github.com/johnwilsoniv/FaceMirror"
#define AppExeName    "S1 Face Mirror.exe"
; PyInstaller --onedir output (relative to this .iss file's directory)
#define AppSourceDir  "dist\S1 Face Mirror"

[Setup]
; A unique GUID for the application — generated once, never changed.
; Inno requires the leading {{ to escape the brace in compiled metadata.
AppId={{C7E4F1A2-9F38-4B2C-8E5A-FACEMIRR0R001}
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

OutputBaseFilename=FaceMirror-S1-{#AppVersion}-win64-cuda121
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern

; 64-bit only — CUDA wheels are x64-only and we ship a 64-bit Python.
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; License + uninstaller
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
; The whole PyInstaller --onedir tree, recursively. recursesubdirs + createallsubdirs
; preserves the layout (which PyInstaller depends on at runtime).
Source: "{#AppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
{
  Pre-install hint: warn the user if no NVIDIA driver is detected. We don't
  hard-block because the app falls back to CPU when CUDA is unavailable.
}
function InitializeSetup(): Boolean;
var
  DriverPath: string;
begin
  Result := True;
  DriverPath := ExpandConstant('{sys}\nvcuda.dll');
  if not FileExists(DriverPath) then
  begin
    if MsgBox('No NVIDIA driver was detected (nvcuda.dll missing from System32). ' +
              'The app will still install and run on CPU, but CUDA acceleration ' +
              'requires an NVIDIA driver supporting CUDA 12.1 or later.' + #13#10 + #13#10 +
              'Continue installing anyway?', mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end;
end;
