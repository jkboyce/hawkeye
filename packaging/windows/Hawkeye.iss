;Inno Setup 5 configuration file for Hawkeye
;
;This file will be executed next to the application bundle image

#define MyAppName "Hawkeye"
#define MyAppVersion "1.3"
#define MyAppYear "2020"
#define MyAppExeName "Hawkeye.exe"
#define MyAppIconsName "Hawkeye.ico"
#define MyWizardImageFileName "Hawkeye-setup-icon.bmp"

[Setup]
AppId={{Hawkeye}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher=Jack Boyce
;AppComments={#MyAppName}
AppCopyright=Copyright (C) {#MyAppYear}
;First option installs per-user, second system-wide
;DefaultDirName={localappdata}\{#MyAppName}
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableStartupPrompt=No
DisableDirPage=Yes
DisableProgramGroupPage=Yes
DisableReadyPage=Yes
DisableFinishedPage=Yes
DisableWelcomePage=No
;Optional License
LicenseFile=
;Windows version requirements for OpenCV, Qt, FFmpeg, Python?
;MinVersion=
OutputBaseFilename={#MyAppName}-{#MyAppVersion}
Compression=lzma
SolidCompression=yes
;First line is for per-user installation, second is system-wide:
;PrivilegesRequired=lowest
PrivilegesRequired=admin
SetupIconFile={#MyAppIconsName}
UninstallDisplayIcon={app}\{#MyAppIconsName}
UninstallDisplayName={#MyAppName}
WizardImageStretch=No
WizardSmallImageFile={#MyWizardImageFileName}
ArchitecturesInstallIn64BitMode=x64


[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "dist\Hawkeye\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\Hawkeye\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#MyAppIconsName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIconsName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Parameters: "-Xappcds:generatecache"; Check: returnFalse()
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,Juggling Lab}"; Flags: nowait postinstall skipifsilent; Check: returnTrue()
Filename: "{app}\{#MyAppExeName}"; Parameters: "-install -svcName ""Hawkeye"" -svcDesc ""Hawkeye"" -mainExe ""{#MyAppExeName}""  "; Check: returnFalse()

[UninstallRun]
Filename: "{app}\{#MyAppExeName} "; Parameters: "-uninstall -svcName Hawkeye -stopOnUninstall"; Check: returnFalse()

[Code]
function returnTrue(): Boolean;
begin
  Result := True;
end;

function returnFalse(): Boolean;
begin
  Result := False;
end;

function InitializeSetup(): Boolean;
begin
// Possible future improvements:
//   if version less or same => just launch app
//   if upgrade => check if same app is running and wait for it to exit
//   Add pack200/unpack200 support?
  Result := True;
end;
