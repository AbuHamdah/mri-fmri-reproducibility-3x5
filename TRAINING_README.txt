FRESH TRAINING PROTOCOL (fresh_supervised_v3)

Use the pinned TensorFlow 2.12 / Python 3.10 environment. All inputs and labels come from INPUTS.json. Splits are validated for complete, disjoint participant sets. Architecture archives supply config.json only; numerical weights and previous target probabilities are never loaded. All network layers are trainable. Gradients use training participants only; validation BCE selects the checkpoint; test predictions are written after selection.

Preflight without training, from the package root in a compatible Python environment:
python source/train_fresh.py --kind mri --seed 42 --fold 1 --output D:/new_mri_run --check-only
python source/train_fresh.py --kind fmri --seed 42 --fold 1 --output D:/new_fmri_run --check-only

To train intentionally, remove --check-only. Defaults: 5 epochs, 1 participant per batch, Adam learning rate 1e-5. --epochs, --batch-size and --lr are explicit controls. Only a new directory outside the package is accepted. Architecture choices currently use the adopted MRI baseline and parent fMRI baseline. This is a new fixed-controller protocol, not a reproduction of every historical comparison or an assertion of matching a paper.

After training both branches for matching seed/fold:
python source/train_fusion_fresh.py --mri D:/new_mri_run --fmri D:/new_fmri_run --output D:/new_fusion_run --seed 42 --fold 1 --check-only
Remove --check-only only to request training. The loader checks protocol version, initialization, modality, exact split and model hash. Fusion normalization uses training rows; alpha uses validation BCE. It writes separate feature and stacking heads plus test predictions. Branch models remain necessary for inference through these heads.

The current changes were checked without any optimizer updates or cohort training. Initialization, label/split validation and fusion prerequisites passed tests; end-to-end training execution is not yet verified. Processed inputs and architecture choices are inherited resources; new initialization alone does not validate their experimental selection or preprocessing.


AUTOMATIC 3 x 5 CROSS-VALIDATION

From PowerShell in the package folder:
.\TRAIN_CV.ps1 -Output D:/NEW_CV -CheckOnly
This checks all 15 partitions and job dependencies without creating output or training.

To intentionally execute the complete new protocol:
.\TRAIN_CV.ps1 -Output D:/NEW_CV -Epochs 5
Default Kind=all: 15 MRI jobs, 15 fMRI jobs and 15 fusion jobs. Each fusion job follows the matching MRI and fMRI jobs. Choose -Kind mri or -Kind fmri for only 15 branch jobs. The output directory must not already exist and must be outside this package. Outputs are under D:/NEW_CV/results. Failed jobs stop the pipeline and write FAILED.json with a log path. No automatic skip or reuse of incomplete outputs occurs.

The pinned Docker image and all training inputs are read-only; the new output directory is the only writable host mount. Training and inference are distinct entry points: TRAIN_CV.ps1 trains new weights; REPLAY.ps1 loads the existing weights and reproduces their stored results.

Seeds: 42,43,44. Folds: 1,2,3,4,5. Before training, every fold is checked for disjoint train/validation/test sets, and each subject must occur in test exactly once per seed. The five held-out folds are pooled within each seed (42 participants); mean and sample SD are computed across the three pooled-seed metric vectors, not by treating 15 folds as independent observations. SUMMARY.json and POOLED_PER_SEED.json are written only after all jobs pass. COMPLETE.json is written last.

In an existing compatible Python environment, the equivalent entry point is:
python source/run_cv.py --output D:/NEW_CV --kind all --check-only
Omit --check-only only to start training; this direct command writes into D:/NEW_CV itself.

Five scheduler/aggregation tests passed, including real 3x5 coverage, missing/duplicate prediction rejection, dependency ordering and stop-on-failure. Docker preflight was run through TRAIN_CV.ps1. No optimizer update or actual cohort training was executed in this launcher update; full training execution remains to be tested when explicitly run.
