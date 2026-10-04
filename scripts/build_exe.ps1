[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$RunTests,
    [switch]$SmokeDemo,
    [switch]$OneFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$File,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )

    Write-Host ("> {0} {1}" -f $File, ($Arguments -join " "))
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $File"
    }
}

function Find-Python {
    foreach ($candidate in @("py", "python")) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }
    throw "Python was not found. Install Python 3.11+ and retry."
}

$python = Find-Python
$pythonArgs = @()
if ($python -eq "py") {
    $pythonArgs = @("-3.11")
}

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    Invoke-Checked -File $python -Arguments ($pythonArgs + $Arguments)
}

if ($Clean) {
    foreach ($target in @((Join-Path $root "build"), (Join-Path $root "dist"))) {
        if (Test-Path -LiteralPath $target) {
            Write-Host "Removing $target"
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $root "pyproject.toml"))) {
    Write-Warning "pyproject.toml is not present yet; continuing so the script can be used during bootstrap."
}

if ($RunTests) {
    Invoke-Python -Arguments @("-m", "pytest", "-q")
}

# Prefer an explicit package entry point. The fallback keeps the script useful while the
# project is being bootstrapped; a missing entry point is reported clearly by PyInstaller.
$entryCandidates = @(
    (Join-Path $root "scripts\launcher.py"),
    (Join-Path $root "src\ai_scientist_mini\__main__.py")
)
$entry = $entryCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $entry) {
    throw "Could not find a package entry point under src\ai_scientist_mini."
}

# PyInstaller is intentionally invoked through the active Python environment; it may be
# installed ahead of time in an offline build image.
$pyInstallerCheck = & $python @pythonArgs -m PyInstaller --version 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed. Install it in the build environment (offline is supported) and retry."
}

$modeArgs = @("--onedir")
if ($OneFile) {
    $modeArgs = @("--onefile")
}

$commonArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--name", "AIScientistMini",
    "--windowed",
    "--exclude-module", "matplotlib",
    "--exclude-module", "numpy",
    "--exclude-module", "pandas",
    "--exclude-module", "PySide6",
    "--paths", (Join-Path $root "src")
) + $modeArgs + @($entry)

Invoke-Python -Arguments $commonArgs

$artifactRoot = Join-Path $root "dist"
$bundleDir = Join-Path $artifactRoot "AIScientistMini"
$exe = if ($OneFile) {
    Join-Path $artifactRoot "AIScientistMini.exe"
} else {
    Join-Path $bundleDir "AIScientistMini.exe"
}

if (-not (Test-Path -LiteralPath $exe)) {
    throw "Build completed but expected executable was not found: $exe"
}

# Include human-readable docs in the distributable directory when using onedir.
if (-not $OneFile) {
    foreach ($doc in @("README.md", "ARCHITECTURE.md", "SAFETY_REPRODUCIBILITY_AUDIT.md", "TEST_STRATEGY.md")) {
        $source = Join-Path $root $doc
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination (Join-Path $bundleDir $doc) -Force
        }
    }
}

Write-Host "Built: $exe"

if ($SmokeDemo) {
    $smokeDir = Join-Path ([System.IO.Path]::GetTempPath()) ("ai-scientist-mini-smoke-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $smokeDir | Out-Null
    try {
        # The executable may expose either `demo memory-strategy` or `demo`; use the
        # documented command first and fail with its output if the CLI contract differs.
        if ($OneFile) {
            & $exe "demo" "memory-strategy" "--data-dir" $smokeDir
        } else {
            & $exe "demo" "memory-strategy" "--data-dir" $smokeDir
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Packaged demo exited with code $LASTEXITCODE"
        }
        # Check the two stable top-level artifacts directly.  This is more
        # reliable on Windows runners than filtering a recursive provider
        # enumeration while the packaged process has just finished writing.
        $reportPath = Join-Path $smokeDir "research_report.md"
        $manifestPath = Join-Path $smokeDir "reproducibility_manifest.json"
        $reports = @(
            @($reportPath, $manifestPath) |
                Where-Object { Test-Path -LiteralPath $_ }
        )
        if ($reports.Length -lt 1) {
            throw "Packaged demo did not produce a report or reproducibility manifest."
        }
        Write-Host "Packaged demo smoke test passed ($($reports.Count) report/manifest files)."
    }
    finally {
        if (Test-Path -LiteralPath $smokeDir) {
            Remove-Item -LiteralPath $smokeDir -Recurse -Force
        }
    }
}
