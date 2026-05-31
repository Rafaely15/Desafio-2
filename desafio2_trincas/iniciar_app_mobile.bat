@echo off
chcp 65001 >nul
setlocal

title Registro de Rachaduras - Mobile

cd /d "%~dp0"

echo.
echo  ============================================
echo   REGISTRO DE RACHADURAS E FISSURAS
echo   Iniciando para acesso via celular...
echo  ============================================
echo.

set "PYTHON=C:\Users\Carla Batista\anaconda3\envs\yolov11\python.exe"
set "CLOUDFLARED=%~dp0cloudflared.exe"
set "PORT=8501"
set "STREAMLIT_OUT=%~dp0streamlit_out.txt"
set "STREAMLIT_ERR=%~dp0streamlit_err.txt"

echo [1/4] Encerrando processos anteriores...
taskkill /F /IM cloudflared.exe >nul 2>&1
taskkill /F /IM streamlit.exe >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Registro de Rachaduras - Streamlit" >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$processIds = Get-NetTCPConnection -LocalPort %PORT% -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($processId in $processIds) { Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 2 /nobreak >nul

if not exist "%PYTHON%" (
    echo.
    echo  ERRO: Python do ambiente yolov11 nao encontrado em:
    echo  %PYTHON%
    echo.
    echo  Solucao: confirme se o ambiente conda yolov11 existe.
    pause
    exit /b 1
)

if not exist "%CLOUDFLARED%" (
    echo.
    echo  ERRO: cloudflared.exe nao encontrado em:
    echo  %CLOUDFLARED%
    echo.
    echo  Solucao: coloque o cloudflared.exe na pasta desafio2_trincas.
    pause
    exit /b 1
)

if not exist "app\app.py" (
    echo.
    echo  ERRO: app\app.py nao encontrado.
    echo  Pasta atual: %CD%
    echo.
    echo  O script precisa estar dentro da pasta desafio2_trincas.
    pause
    exit /b 1
)

echo [2/4] Iniciando Streamlit na porta %PORT%...
echo Logs do Streamlit: "%STREAMLIT_OUT%" e "%STREAMLIT_ERR%"

break > "%STREAMLIT_OUT%"
break > "%STREAMLIT_ERR%"

start "Registro de Rachaduras - Streamlit" /MIN cmd /c ""%PYTHON%" -m streamlit run app\app.py --server.address 127.0.0.1 --server.port %PORT% --server.headless true --server.enableCORS false --server.enableXsrfProtection false 1>"%STREAMLIT_OUT%" 2>"%STREAMLIT_ERR%""

echo [3/4] Aguardando Streamlit iniciar...
set /a tentativas=0

:aguarda
set /a tentativas+=1
timeout /t 3 /nobreak >nul

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$r = try { Invoke-WebRequest 'http://127.0.0.1:%PORT%/_stcore/health' -UseBasicParsing -TimeoutSec 5 } catch { $null }; if ($r -and $r.StatusCode -eq 200) { exit 0 } else { exit 1 }" >nul 2>&1

if %ERRORLEVEL%==0 goto :tunel
if %tentativas% LSS 10 goto :aguarda

echo.
echo  ERRO: Streamlit nao respondeu apos 30 segundos.
echo.
echo  Veja os detalhes em:
echo  %STREAMLIT_ERR%
echo  %STREAMLIT_OUT%
echo.
type "%STREAMLIT_ERR%"
pause
exit /b 1

:tunel
echo     Streamlit OK na porta %PORT%!
echo.
echo [4/4] Criando tunel publico...
echo.
echo  ============================================
echo   AGUARDE O LINK trycloudflare.com abaixo.
echo.
echo   ^>^> Copie o link e abra no celular ^<^<
echo   ^>^> Cada vez que iniciar, o link muda ^<^<
echo   ^>^> Ctrl+C encerra o tunel            ^<^<
echo  ============================================
echo.
echo  Gerando link... aguarde alguns segundos...
echo.

"%CLOUDFLARED%" tunnel --url http://127.0.0.1:%PORT%

echo.
echo  Tunel encerrado.
pause >nul
