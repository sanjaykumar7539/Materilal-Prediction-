@echo off
setlocal
set "OWNLLM_DOMAIN=materials"
set "OWNLLM_MATERIALS_KB=%~dp0..\domains\materials\verified_records.json"
set "OWNLLM_MATERIALS_FEEDBACK=%~dp0..\data\materials_feedback.sqlite3"
set "OWNLLM_MODEL_PATH=%~dp0..\runs\chemistry_properties_v4\export\ownllm.pt"
set "TOKENIZER_PATH=%~dp0..\runs\chemistry_properties_v4\export\tokenizer.json"
set "DEVICE=cpu"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "OWNLLM_PYTHON=%~dp0..\.tools\qa_python\Scripts\python.exe"
if not exist "%OWNLLM_PYTHON%" set "OWNLLM_PYTHON=python"
if not exist "%OWNLLM_MODEL_PATH%" exit /b 1
if not exist "%TOKENIZER_PATH%" exit /b 1
pushd "%~dp0.."
"%OWNLLM_PYTHON%" -m uvicorn ownllm.api.service:app --app-dir "%~dp0..\src" --host 127.0.0.1 --port 8000
popd
