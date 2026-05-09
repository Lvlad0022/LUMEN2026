param(
    [string]$ModelPath = "",
    [int]$Port = 8000,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

if ($ModelPath -ne "") {
    $env:MODEL_PATH = $ModelPath
}

$env:PYTHONPATH = ".;classic_methods/src"

& .\classic_methods\.venv\Scripts\python.exe -m uvicorn `
    recommendation_api:app `
    --host $HostAddress `
    --port $Port
