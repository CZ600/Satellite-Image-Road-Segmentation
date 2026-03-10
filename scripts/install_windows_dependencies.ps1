param(
    [string]$CondaExe = "conda"
)

$ErrorActionPreference = "Stop"

& $CondaExe install -y pytorch torchvision pytorch-cuda=12.4 tensorboard gdal scikit-image scikit-learn numba -c pytorch -c nvidia -c conda-forge
python -m pip install opencv-python sknw chardet tqdm
