@echo off
chcp 65001 >nul
title Registro de Rachaduras — Mobile

echo.
echo  ============================================
echo   REGISTRO DE RACHADURAS E FISSURAS
echo   Iniciando para acesso via celular...
echo  ============================================
echo.

set PYTHON=C:\Users\Carla Batista\anaconda3\envs\yolov11\python.exe
set STREAMLIT=C:\Users\Carla Batista\anaconda3\envs\yolov11\Scripts\streamlit.exe
set PORT=8501

:: ── 1. Garante que nao ha processos anteriores ────────────────────────────────
echo [1/4] Encerrando processos anteriores...
taskkill /F /IM cloudflared.exe >nul 2>&1
taskkill /F /IM streamlit.exe   >nul 2>&1
timeout /t 2 /nobreak >nul

:: ── 2. Verifica se Python do yolov11 existe ───────────────────────────────────
if not exist "%STREAMLIT%" (
    echo.
    echo  ERRO: Streamlit nao encontrado em:
    echo  %STREAMLIT%
    echo.
    echo  Solucao: abra o Anaconda Prompt e execute:
    echo     conda activate yolov11
    echo     pip install streamlit
    echo.
    pause & exit /b 1
)

:: ── 3. Inicia Streamlit na porta 8501 ─────────────────────────────────────────
echo [2/4] Iniciando Streamlit na porta %PORT%...
start /B "" "%STREAMLIT%" run app\app.py ^
    --server.address 127.0.0.1 ^
    --server.port %PORT% ^
    --server.headless true ^
    --server.enableCORS false ^
    --server.enableXsrfProtection false

:: ── 4. Aguarda Streamlit ficar pronto (tenta por até 30s) ─────────────────────
echo [3/4] Aguardando Streamlit iniciar...
set /a tentativas=0
:aguarda
set /a tentativas+=1
timeout /t 3 /nobreak >nul
powershell -Command ^
  "$r = try { Invoke-WebRequest 'http://localhost:%PORT%/_stcore/health' -UseBasicParsing -TimeoutSec 2 } catch { $null }; if ($r.Content -eq 'ok') { exit 0 } else { exit 1 }" >nul 2>&1
if %ERRORLEVEL%==0 goto :tunel
if %tentativas% LSS 10 goto :aguarda

echo.
echo  ERRO: Streamlit nao respondeu apos 30 segundos.
echo  Verifique se o ambiente yolov11 esta correto.
pause & exit /b 1

:: ── 5. Inicia tunel Cloudflare apontando para porta 8501 ─────────────────────
:tunel
echo     Streamlit OK na porta %PORT%!
echo.
echo [4/4] Criando tunel publico...
echo.
echo  ============================================
echo   AGUARDE O LINK trycloudflare.com abaixo.
echo.
echo   >> Copie o link e abra no celular <<
echo   >> Cada vez que iniciar, o link muda <<
echo   >> Ctrl+C encerra o tunel            <<
echo  ============================================
echo.
echo  Gerando link... aguarde alguns segundos...
echo.

cloudflared.exe tunnel --url http://localhost:%PORT%

echo.
echo  Tunel encerrado.
pause >nul
