# Current stage: model deployment usage.
# Starts the local MiniCPM FastAPI service in the background.

param(
    [string]$AdapterDir = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $env:USERPROFILE "miniconda3\envs\self-llm\python.exe"
$OutLog = Join-Path $PSScriptRoot "minicpm_api.out.log"
$ErrLog = Join-Path $PSScriptRoot "minicpm_api.err.log"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

$existing = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "API service is already listening on port 8000. PID: $($existing.OwningProcess)"
    exit 0
}

if ($AdapterDir) {
    $ResolvedAdapterDir = (Resolve-Path -LiteralPath $AdapterDir).Path
    $env:MINICPM_ADAPTER_DIR = $ResolvedAdapterDir
    Write-Host "Adapter: $ResolvedAdapterDir"
} else {
    Remove-Item Env:MINICPM_ADAPTER_DIR -ErrorAction SilentlyContinue
}

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @("-m", "uvicorn", "local_examples.minicpm_api:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started MiniCPM API. PID: $($process.Id)"
Write-Host "Docs: http://127.0.0.1:8000/docs"
Write-Host "WebDemo: http://127.0.0.1:8000/"
