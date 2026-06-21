# Current stage: model deployment usage.
# Shows the local API port, process, recent logs, and GPU memory status.

$ErrorActionPreference = "Continue"

Write-Host "== Port 8000 =="
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess |
    Format-Table -AutoSize

Write-Host "`n== Python processes =="
Get-Process python -ErrorAction SilentlyContinue |
    Select-Object Id, ProcessName, Path |
    Format-Table -AutoSize

Write-Host "`n== GPU =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

Write-Host "`n== API stderr log tail =="
$ErrLog = Join-Path $PSScriptRoot "minicpm_api.err.log"
if (Test-Path -LiteralPath $ErrLog) {
    Get-Content -LiteralPath $ErrLog -Tail 40
} else {
    Write-Host "No stderr log found."
}
