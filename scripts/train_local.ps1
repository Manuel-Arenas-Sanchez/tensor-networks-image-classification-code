param(
    [string]$Config = "configs/experiments/mnist_mps.yaml",
    [string]$Runtime = "configs/runtime/local.yaml",
    [string]$Device = "auto"
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ResolvedConfig = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $ProjectRoot $Config }
$ResolvedRuntime = if ([System.IO.Path]::IsPathRooted($Runtime)) { $Runtime } else { Join-Path $ProjectRoot $Runtime }

& $Python -m tn_dl.cli.train --config $ResolvedConfig --runtime $ResolvedRuntime --device $Device
