# Current stage: model deployment usage.
# Stops the local MiniCPM FastAPI service by finding the process on port 8000.

$ErrorActionPreference = "Stop"

$connections = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not $connections) {
    Write-Host "No API service is listening on port 8000."
    exit 0
}

$pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pidValue in $pids) {
    Stop-Process -Id $pidValue -Force
    Write-Host "Stopped MiniCPM API process. PID: $pidValue"
}
