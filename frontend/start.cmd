@echo off
setlocal
set "CHAT_NODE=%~dp0..\.tools\node-v22.14.0-win-x64\node.exe"
if exist "%CHAT_NODE%" goto run
where node >nul 2>nul
if errorlevel 1 (
  echo Install Node.js 22 or newer, then run npm install and npm run build in frontend.
  exit /b 1
)
set "CHAT_NODE=node"
:run
if not exist "%~dp0dist\index.html" (
  echo Build the frontend first: run npm install and npm run build in frontend.
  exit /b 1
)
"%CHAT_NODE%" "%~dp0server.js"
