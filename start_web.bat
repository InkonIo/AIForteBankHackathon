@echo off
REM ML Fraud Detection - Web Interface Launcher (Windows)

echo =========================================
echo 🚀 ML Fraud Detection - Web Interface
echo =========================================
echo.

REM Проверка ML API сервера
echo 📡 Проверка ML API сервера...
curl -s http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ ML API сервер работает на http://localhost:5000
) else (
    echo ❌ ML API сервер не запущен!
    echo.
    echo Пожалуйста, запустите сервер в другом окне:
    echo    python ml_service_improved.py
    echo.
    pause
)

echo.
echo 🌐 Запуск веб-интерфейса...
echo.
echo =========================================
echo   Откройте браузер:
echo   http://localhost:8080
echo =========================================
echo.
echo Нажмите Ctrl+C для остановки сервера
echo.

REM Запуск HTTP сервера
python -m http.server 8080