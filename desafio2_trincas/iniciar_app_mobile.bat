@echo off
chcp 65001 >nul
title Registro de Rachaduras — Acesso Mobile

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║     REGISTRO DE RACHADURAS E FISSURAS           ║
echo ║           Acesso via Celular / Tablet            ║
echo ╚══════════════════════════════════════════════════╝
echo.

:: ── Define caminhos ──────────────────────────────────────────────────────────
set PYTHON=C:\Users\Carla Batista\anaconda3\envs\yolov11\python.exe
set STREAMLIT=C:\Users\Carla Batista\anaconda3\envs\yolov11\Scripts\streamlit.exe
set APP=app\app.py
set PORT=8501

:: ── Verifica Python do ambiente yolov11 ──────────────────────────────────────
if not exist "%PYTHON%" (
    echo [ERRO] Python do ambiente yolov11 nao encontrado em:
    echo        %PYTHON%
    echo.
    echo Execute:  conda activate yolov11
    echo e tente novamente.
    pause & exit /b 1
)

:: ── Verifica cloudflared ──────────────────────────────────────────────────────
if not exist cloudflared.exe (
    echo [1/3] Baixando cloudflared...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
    if %ERRORLEVEL% NEQ 0 (
        echo ERRO ao baixar cloudflared. Verifique a conexao.
        pause & exit /b 1
    )
    echo     Download concluido.
) else (
    echo [1/3] cloudflared.exe encontrado. OK
)

:: ── Encerra instâncias anteriores do Streamlit ───────────────────────────────
echo [2/3] Iniciando Streamlit ^(ambiente yolov11^)...
taskkill /F /IM streamlit.exe >nul 2>&1
taskkill /F /IM cloudflared.exe >nul 2>&1
timeout /t 1 /nobreak >nul

:: Inicia Streamlit em background com o ambiente correto
start /B "" "%STREAMLIT%" run "%APP%" ^
    --server.address 127.0.0.1 ^
    --server.port %PORT% ^
    --server.headless true ^
    --server.enableCORS false ^
    --server.enableXsrfProtection false

echo     Aguardando Streamlit iniciar...
timeout /t 6 /nobreak >nul

:: ── Verifica se Streamlit subiu ───────────────────────────────────────────────
powershell -Command "try { $r = Invoke-WebRequest 'http://localhost:%PORT%/_stcore/health' -UseBasicParsing -TimeoutSec 5; if($r.Content -eq 'ok'){Write-Host '[OK] Streamlit rodando.'} } catch { Write-Host '[AVISO] Streamlit ainda iniciando...' }"

:: ── Inicia túnel Cloudflare ───────────────────────────────────────────────────
echo.
echo [3/3] Criando tunel publico Cloudflare...
echo.
echo ╔══════════════════════════════════════════════════╗
echo ║  Copie o link  trycloudflare.com  abaixo        ║
echo ║  e abra no navegador do celular                  ║
echo ║                                                  ║
echo ║  O link muda a cada vez que voce inicia.        ║
echo ║  Pressione Ctrl+C para encerrar tudo.           ║
echo ╚══════════════════════════════════════════════════╝
echo.

cloudflared.exe tunnel --url http://localhost:%PORT%

echo.
echo Tunel encerrado. Pressione qualquer tecla para fechar.
pause >nul
