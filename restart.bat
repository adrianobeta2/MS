@echo off
:: Verifica se está rodando como administrador
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Reabrindo como Administrador...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo ===============================
echo Reiniciando servico SVE
echo ===============================

net stop SVE
timeout /t 2 >nul
net start SVE

echo ===============================
echo Servico SVE reiniciado
echo ===============================
