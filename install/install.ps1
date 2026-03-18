param(
  [ValidateSet("npm", "git")]
  [string]$InstallMethod = "npm",
  [string]$Version = "latest",
  [string]$Package,
  [string]$Repo = "https://github.com/daisyluvr42/Marv.git",
  [string]$Ref = "main",
  [switch]$NoOnboard,
  [switch]$Onboard,
  [switch]$Beta,
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PackageName = if ($env:MARV_PACKAGE_NAME) { $env:MARV_PACKAGE_NAME } else { "agentmarv" }
if ($env:MARV_INSTALL_METHOD) { $InstallMethod = $env:MARV_INSTALL_METHOD }
if ($env:MARV_VERSION) { $Version = $env:MARV_VERSION }
if ($env:MARV_PACKAGE) { $Package = $env:MARV_PACKAGE }
if ($env:MARV_REPO) { $Repo = $env:MARV_REPO }
if ($env:MARV_REF) { $Ref = $env:MARV_REF }
if ($env:MARV_NO_ONBOARD -eq "1") { $NoOnboard = $true }
if ($env:MARV_DRY_RUN -eq "1") { $DryRun = $true }
if ($Beta) { $Version = "beta" }

function Show-Usage {
  @"
Marv installer (Windows PowerShell)

Usage:
  install.ps1 [-InstallMethod npm|git] [-Version <tag>] [-Package <path-or-url>] [-Repo <url>] [-Ref <git-ref>] [-NoOnboard] [-Onboard] [-DryRun]

Examples:
  iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1 | iex
  & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -Beta
  & ([scriptblock]::Create((iwr -useb https://raw.githubusercontent.com/daisyluvr42/Marv/main/install/install.ps1))) -Package .\agentmarv-2026.3.15.tgz
"@
}

if ($args -contains "-h" -or $args -contains "--help") {
  Show-Usage
  exit 0
}

function Invoke-CommandOrDryRun {
  param(
    [Parameter(Mandatory = $true)]
    [scriptblock]$Script,
    [Parameter(Mandatory = $true)]
    [string]$Display
  )

  if ($DryRun) {
    Write-Host "[dry-run] $Display"
    return
  }
  & $Script
}

function Get-NodeMajor {
  if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    return 0
  }
  $nodeVersion = (& node -v).Trim()
  if ($nodeVersion -match '^v(\d+)') {
    return [int]$Matches[1]
  }
  return 0
}

function Ensure-Node22 {
  if ((Get-NodeMajor) -ge 22) {
    return
  }

  if (Get-Command winget -ErrorAction SilentlyContinue) {
    Invoke-CommandOrDryRun -Display "winget install OpenJS.NodeJS.LTS" -Script {
      & winget install --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
    }
  } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
    Invoke-CommandOrDryRun -Display "choco install nodejs-lts -y" -Script {
      & choco install nodejs-lts -y
    }
  } elseif (Get-Command scoop -ErrorAction SilentlyContinue) {
    Invoke-CommandOrDryRun -Display "scoop install nodejs-lts" -Script {
      & scoop install nodejs-lts
    }
  } else {
    throw "Node.js 22+ is required. Install Node.js first, then rerun the installer."
  }
}

function Ensure-Npm {
  if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm is required. Reopen PowerShell after installing Node.js, then rerun the installer."
  }
}

function Resolve-InstallSpec {
  if ($Package) {
    return $Package
  }
  if ($InstallMethod -eq "git") {
    if ($Repo.StartsWith("git+")) {
      return "$Repo#$Ref"
    }
    return "git+$Repo#$Ref"
  }
  return "$PackageName@$Version"
}

function Resolve-MarvCommand {
  $cmd = Get-Command marv -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }
  $npmPrefix = (& npm prefix -g).Trim()
  $candidate = Join-Path $npmPrefix "marv.cmd"
  if (Test-Path $candidate) {
    return $candidate
  }
  return $null
}

Ensure-Node22
Ensure-Npm

$installSpec = Resolve-InstallSpec
Write-Host "Installing Marv from: $installSpec"
$env:SHARP_IGNORE_GLOBAL_LIBVIPS = if ($env:SHARP_IGNORE_GLOBAL_LIBVIPS) { $env:SHARP_IGNORE_GLOBAL_LIBVIPS } else { "1" }
$env:NPM_CONFIG_FUND = "false"
$env:NPM_CONFIG_AUDIT = "false"

Invoke-CommandOrDryRun -Display "npm install -g $installSpec" -Script {
  & npm install -g $installSpec
}

$marvCmd = Resolve-MarvCommand
if (-not $marvCmd) {
  throw "Install finished, but the marv CLI was not found on PATH. Reopen PowerShell, then run: marv --version"
}

Write-Host "CLI ready: $marvCmd"
if (-not $DryRun) {
  & $marvCmd --version
}

$runOnboard = $true
if ($NoOnboard) { $runOnboard = $false }
if ($Onboard) { $runOnboard = $true }

if (-not $runOnboard) {
  Write-Host "Install complete."
  Write-Host "Run this when ready:"
  Write-Host "  $marvCmd onboard --install-daemon"
  exit 0
}

Write-Host "Running onboarding..."
Invoke-CommandOrDryRun -Display "$marvCmd onboard --install-daemon" -Script {
  & $marvCmd onboard --install-daemon
}
