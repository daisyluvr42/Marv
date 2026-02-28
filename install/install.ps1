param(
  [string]$Repo = "https://github.com/daisyluvr42/Marv.git",
  [string]$Ref = "main",
  [switch]$NoOnboard,
  [switch]$Onboard,
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
  @"
Marv installer (Windows PowerShell)

Usage:
  install.ps1 [-Repo <url>] [-Ref <git-ref>] [-NoOnboard] [-Onboard] [-DryRun]

Options:
  -Repo        Git repository URL (default: https://github.com/daisyluvr42/Marv.git)
  -Ref         Git ref/branch/tag (default: main)
  -NoOnboard   Skip `marv onboard --install-daemon`
  -Onboard     Force onboarding after install
  -DryRun      Print commands only
"@
}

if ($args -contains "-h" -or $args -contains "--help") {
  Show-Usage
  exit 0
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  throw "Node.js is required (22+). Please install Node first."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  throw "npm is required. Please install npm first."
}

$nodeVersion = (& node -v).Trim()
if ($nodeVersion -notmatch '^v(\d+)') {
  throw "Unable to parse Node.js version: $nodeVersion"
}

$nodeMajor = [int]$Matches[1]
if ($nodeMajor -lt 22) {
  throw "Node.js 22+ is required. Current version: $nodeVersion"
}

$pkgSpec = if ($Repo.StartsWith("git+")) { "$Repo#$Ref" } else { "git+$Repo#$Ref" }
$installCmd = "npm install -g `"$pkgSpec`""

Write-Host "Installing Marv from: $pkgSpec"
if ($DryRun) {
  Write-Host "[dry-run] $installCmd"
} else {
  & npm install -g $pkgSpec
}

$runOnboard = $true
if ($NoOnboard) { $runOnboard = $false }
if ($Onboard) { $runOnboard = $true }

if (-not $runOnboard) {
  Write-Host "Install complete. Run this when ready:"
  Write-Host "  marv onboard --install-daemon"
  exit 0
}

$marvCmd = "marv"
$npmPrefix = (& npm prefix -g).Trim()
$marvCmdPath = Join-Path $npmPrefix "marv.cmd"
if (Test-Path $marvCmdPath) {
  $marvCmd = $marvCmdPath
}

Write-Host "Running onboarding..."
if ($DryRun) {
  Write-Host "[dry-run] $marvCmd onboard --install-daemon"
} else {
  & $marvCmd onboard --install-daemon
}
