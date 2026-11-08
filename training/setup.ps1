$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot
# Keep downloaded Python, package caches, and the environment local to this project.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.python'
$env:UV_CACHE_DIR = Join-Path $repoRoot '.cache/uv'
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Setup command failed with exit code $LASTEXITCODE" } }
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'uv is required to create the Python 3.12 environment. Install it from https://docs.astral.sh/uv/getting-started/installation/ and rerun.'
    }
    uv python install 3.12
    Check-Exit
    uv venv --python 3.12 .venv
    Check-Exit
}
# CUDA 12.8 wheels support the RTX 50 series; a separate CUDA toolkit is not needed.
& .venv/Scripts/python.exe -m ensurepip --upgrade
Check-Exit
$env:PIP_CACHE_DIR = Join-Path $repoRoot '.cache/pip'
& .venv/Scripts/python.exe -m pip install 'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu128
Check-Exit
& .venv/Scripts/python.exe -m pip install -r training/requirements.txt
Check-Exit
& .venv/Scripts/python.exe -c "import sys; sys.path.insert(0, 'training'); from common import require_cuda; import torch; require_cuda(); print('GPU ready:', torch.cuda.get_device_name(0))"
Check-Exit
Write-Host 'Setup complete. Start training with: .venv/Scripts/python.exe training/train.py'
