@echo off
title fufuchat-demo
cd /d "%~dp0"
py -3 service/CLI.py
if errorlevel 1 echo.
pause