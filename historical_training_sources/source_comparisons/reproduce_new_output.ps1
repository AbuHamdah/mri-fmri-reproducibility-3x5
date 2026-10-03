param([Parameter(Mandatory=$true)][string]$NewOutput)
$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$outputRoot = [System.IO.Path]::GetFullPath($NewOutput)
if (Test-Path -LiteralPath $outputRoot) { throw 'Output folder already exists. Choose a new folder to preserve existing artifacts.' }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $packageRoot 'PLAN.json') -Destination (Join-Path $outputRoot 'PLAN.json')
Copy-Item -LiteralPath (Join-Path $packageRoot 'source') -Destination (Join-Path $outputRoot 'source') -Recurse
foreach ($part in @('inputs','initial_models','reference_splits')) { Copy-Item -LiteralPath (Join-Path $packageRoot $part) -Destination (Join-Path $outputRoot $part) -Recurse }
foreach ($part in @('INPUTS.json','README_EN.md','INPUT_SOURCE_COPY_MANIFEST.json','SOURCE_MODELS_REFERENCE.json','SOURCE_ENCODER_EQUIVALENCE.json','RUNTIME_ENVIRONMENT.json','requirements.txt')) { Copy-Item -LiteralPath (Join-Path $packageRoot $part) -Destination (Join-Path $outputRoot $part) }
$runtimeArgs = @('run','--rm','--cpus','2','--memory','8g','-e','OMP_NUM_THREADS=2','-e','TF_NUM_INTRAOP_THREADS=2','-e','TF_NUM_INTEROP_THREADS=2','-e','CUDA_VISIBLE_DEVICES=-1','-v',($packageRoot + ':/package:ro'),'-v',($outputRoot + ':/out'),'dyslexia-pipeline-tf212-py310:latest')
foreach ($step in @('train_branches.py','train_fusion.py','audit_delivery.py','plot_delivery.py')) {
    $stepArgs = $runtimeArgs + @('python','-u',('/out/source/' + $step))
    $logName = switch ($step) { 'train_branches.py' { 'training.log' }; 'train_fusion.py' { 'fusion_training.log' }; 'audit_delivery.py' { 'final_audit.log' }; default { 'plot_delivery.log' } }
    & docker @stepArgs 2>&1 | Tee-Object -FilePath (Join-Path $outputRoot $logName)
    if ($LASTEXITCODE -ne 0) { throw ('Step failed and artifacts were preserved: ' + $step) }
}
Write-Output ('Completed inherited last-layer reproduction in ' + $outputRoot)
Write-Output 'This is not independent cross-validation or fresh whole-network from-scratch training.'
