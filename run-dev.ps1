$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = Join-Path $root 'desktop'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $desktop)) {
    throw "Desktop app folder not found: $desktop"
}

if (-not (Test-Path $venvPython)) {
    throw "Python interpreter not found: $venvPython"
}

$frontendCommand = "Set-Location '$desktop'; npm install; npm run tauri:dev"

Start-Process -FilePath 'powershell.exe' -ArgumentList @(
    '-NoLogo',
    '-NoExit',
    '-ExecutionPolicy',
    'Bypass',
    '-Command',
    $frontendCommand
)

Set-Location $root
& $venvPython -m scripts.serve