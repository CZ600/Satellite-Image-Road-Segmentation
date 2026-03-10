param(
    [string]$PythonExe = "python",
    [string]$SourceDeepGlobeRoot = "D:\project\pythonProject\Road_Identification\SAM2-UNet\deepglobe",
    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not $SkipPrepare) {
    & $PythonExe scripts/prepare_deepglobe.py --source-root $SourceDeepGlobeRoot --target-root Datasets/DeepGlobe --crop-size 512
}

& $PythonExe train.py -m ConvNeXt_UPerNet_DGCN_MTL -d DeepGlobe
