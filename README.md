# MRI/fMRI reproducibility and 3 × 5 CV

Private research workspace containing saved-model inference, 33 result rows, comparison tables, and a separate fresh-training implementation.

## Restore the complete package

The repository holds code, results, manifests and documentation. Model files, processed participant arrays and the pinned Docker environment are packaged as release assets to keep Git manageable. All binary contents are preserved exactly; repeated chunks are stored once.

```powershell
git clone https://github.com/AbuHamdah/mri-fmri-reproducibility-3x5.git
cd mri-fmri-reproducibility-3x5
gh auth login
python restore_assets.py
```

Requires Python 3.10+ and GitHub CLI. Access to this private repository is required. The restoration script verifies archive, chunk and reconstructed-file SHA256 hashes and refuses to overwrite different existing files. Allow at least 35 GB of free disk space for downloads and restored data.

## Reproduce saved results

With Docker installed, run `./REPLAY.ps1` in PowerShell. It uses the included pinned environment, disables networking during execution, and performs inference without retraining. The completed verification covers 197 full-model inference jobs, 495 result/model records and all 33 reported results. Classification-derived metrics matched exactly; continuous metrics matched within the original tolerances. See `reproduction_run/REPRODUCTION_AUDIT.json` and `reproduction_run/PIPELINE_AUDIT.json`.

View `START_RESULTS_EN.html` for the results, or read `RESULTS.json`.

## New training

```powershell
# Preflight only; does not train:
./TRAIN_CV.ps1 -Output D:/NEW_CV -CheckOnly

# Start training intentionally into a new external directory:
./TRAIN_CV.ps1 -Output D:/NEW_CV -Epochs 5
```

The new scheduler uses seeds 42, 43 and 44 with five folds each: 15 MRI jobs, 15 fMRI jobs and 15 fusion jobs. It pools held-out predictions within each seed, then computes the mean and sample SD across seeds. This fresh-training protocol was added later and did not generate the stored results. Its scheduler and initialization checks passed; a complete new training run has not been executed. See `TRAINING_README.txt` for settings and scope.

Saved-checkpoint reproduction verifies execution and numerical agreement. It does not establish independent experimental validation or reconstruct the original training history. This package contains processed arrays, not the complete raw preprocessing workflow. Treat participant-level files as private research data.
