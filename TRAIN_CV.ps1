param(
    [Parameter(Mandatory = $true)][string]$Output,
    [ValidateSet('mri','fmri','all')][string]$Kind = 'all',
    [ValidateRange(1,100000)][int]$Epochs = 5,
    [ValidateRange(1,10000)][int]$BatchSize = 1,
    [double]$LearningRate = 0.00001,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$taskOutput = [IO.Path]::GetFullPath($Output)
if ((Test-Path -LiteralPath $taskOutput) -or $taskOutput.Equals($taskRoot, [StringComparison]::OrdinalIgnoreCase) -or $taskOutput.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output must be a new directory outside the package.'
}
if ([double]::IsNaN($LearningRate) -or [double]::IsInfinity($LearningRate) -or $LearningRate -le 0) { throw 'LearningRate must be positive and finite.' }
$taskImage = 'sha256:1afede9925046727d60d3a374ecbe32fe6984326a7aafc08167832a7d9fb307b'
& docker image inspect $taskImage *> $null
if ($LASTEXITCODE -ne 0) {
    & docker image load --input (Join-Path $taskRoot 'environment/tf212_py310_image.tar')
    if ($LASTEXITCODE -ne 0) { throw 'Unable to load the supplied environment.' }
}
$taskDockerArgs = @('run','--rm','--network','none','--cpus','2','--memory','8g',
    '-e','CUDA_VISIBLE_DEVICES=-1','-e','TF_NUM_INTRAOP_THREADS=2','-e','TF_NUM_INTEROP_THREADS=2',
    '-e','PYTHONDONTWRITEBYTECODE=1','--mount',"type=bind,source=$taskRoot,target=/package,readonly")
if (-not $CheckOnly) {
    New-Item -ItemType Directory -Path $taskOutput | Out-Null
    $taskDockerArgs += @('--mount',"type=bind,source=$taskOutput,target=/output")
}
$taskDockerArgs += @($taskImage,'python','/package/source/run_cv.py','--output','/output/results',
    '--kind',$Kind,'--epochs',"$Epochs",'--batch-size',"$BatchSize",'--lr',$LearningRate.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($CheckOnly) { $taskDockerArgs += '--check-only' }
& docker @taskDockerArgs
if ($LASTEXITCODE -ne 0) { throw 'CV command failed. If training started, inspect results/FAILED.json and results/logs in the output directory.' }
if ($CheckOnly) { Write-Host '3 x 5 preflight passed; no training started.' }
else { Write-Host "Completed. Results: $taskOutput\results" }
