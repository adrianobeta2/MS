# Verifica se está como administrador
if (-not ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {

    Write-Host "Reabrindo como Administrador..."
    Start-Process powershell `
        -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" `
        -Verb RunAs
    exit
}

Write-Host "==============================="
Write-Host "Reiniciando servico SVE"
Write-Host "==============================="

Stop-Service -Name "SVE" -Force
Start-Sleep -Seconds 2
Start-Service -Name "SVE"

Write-Host "==============================="
Write-Host "Servico SVE reiniciado"
Write-Host "==============================="
