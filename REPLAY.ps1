$ErrorActionPreference = "Stop"
$taskRoot = $PSScriptRoot
$taskImage = "sha256:1afede9925046727d60d3a374ecbe32fe6984326a7aafc08167832a7d9fb307b"
& docker image inspect $taskImage *> $null
if ($LASTEXITCODE -ne 0) {
    & docker image load --input (Join-Path $taskRoot "environment/tf212_py310_image.tar")
    if ($LASTEXITCODE -ne 0) { throw "Unable to load the included reproduction environment." }
}
& docker run --rm --network none --cpus 16 --memory 24g -e CUDA_VISIBLE_DEVICES=-1 -e TF_CPP_MIN_LOG_LEVEL=2 -e PYTHONPATH=/package/source --mount "type=bind,source=$taskRoot,target=/package" $taskImage python /package/source/run_reproduction.py
if ($LASTEXITCODE -ne 0) { throw "Reproduction failed. Inspect reproduction_run/FAILURE.json and EXECUTION.log." }
Write-Host "Reproduction passed. Read reproduction_run/REPRODUCTION_AUDIT.json."
