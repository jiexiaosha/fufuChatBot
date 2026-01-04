REM ...existing code...
@echo off
title fufuchat-demo
cd /d "%~dp0"
setlocal enabledelayedexpansion

REM 使用本地编码（GBK），避免 UTF-8 BOM 导致的 '' 错误
chcp 65001 >nul

REM 说明：输入 conda 环境名（回车为 conda base），或输入本地 venv/.venv 文件夹名
set /p ENV_NAME=环境名(回车为 conda base 或输入本地 venv 文件夹名):

if "%ENV_NAME%"=="" (
  call :try_conda_activate base
  goto RUN
)

REM 优先检测项目下 venv/.venv
if exist "%~dp0%ENV_NAME%\Scripts\activate.bat" (
  call "%~dp0%ENV_NAME%\Scripts\activate.bat"
  echo 已激活本地虚拟环境: %ENV_NAME%
  goto RUN
)

REM 否则尝试按 conda 环境名激活
call :try_conda_activate "%ENV_NAME%"

:RUN
REM 👇 关键新增：使用 Hugging Face 国内镜像
set HF_ENDPOINT=https://hf-mirror.com
set HF_TOKEN=
python -m service.CLI
if errorlevel 1 echo.
pause
exit /b 0

:try_conda_activate
set "ENV_TO_ACT=%~1"
set "CONDA_ACTIVATED="

REM 常见 conda 安装路径
for %%p in ("%USERPROFILE%\anaconda3\condabin\conda.bat" "%USERPROFILE%\miniconda3\condabin\conda.bat" "C:\ProgramData\Anaconda3\condabin\conda.bat") do (
  if exist "%%~p" (
    call "%%~p" activate %ENV_TO_ACT%
    set "CONDA_ACTIVATED=1"
    goto conda_done
  )
)

REM 尝试 PATH 中的 conda
where conda >nul 2>&1
if not errorlevel 1 (
  call conda activate %ENV_TO_ACT% 2>nul
  if not errorlevel 1 set "CONDA_ACTIVATED=1"
)

:conda_done
if defined CONDA_ACTIVATED (
  echo 已激活 conda 环境: %ENV_TO_ACT%
) else (
  echo 未检测到可用的 conda 激活脚本，未激活任何 conda 环境。
)
exit /b 0