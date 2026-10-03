RESULTS AND COMPARISONS



The package contains 33 result rows: 7 adopted results, 10 source comparisons and 16 fusion comparisons. Values and predictions are unchanged.



REPLAY

Run REPLAY.ps1 with Docker installed. It loads the included pinned TensorFlow 2.12 image if necessary, performs fresh saved-model inference and recomputes the results. The package supplies processed arrays; raw-data preprocessing is not rerun. The standard entry point uses 16 CPUs and a 24 GB Docker memory limit.



Use the supplied TensorFlow 2/2 thread configuration and existing tolerances. source/replay.py can reuse a validated cache; REPLAY.ps1 starts a fresh verification. Original execution evidence is retained under evidence and preserved_attempts.



NEW TRAINING

See TRAINING_README.txt. source/train_fresh.py initializes every network weight afresh and uses recorded labels. source/train_fusion_fresh.py accepts only matching new branch outputs. New outputs must be outside this package. Historical training entry points are disabled. The new protocol has not been trained on this cohort and does not replace any of the 33 results or reproduce their historical training procedures. Its controller is fixed; no new LLM operation is claimed.



VERIFICATION SCOPE

Checks cover file integrity and saved-model inference.  The three seeds concern the same 42 participants. Full execution of the new training protocol remains untested.



MANIFESTS

FINAL_PACKAGE_INVENTORY.json describes current file hashes. Copy manifests preserve original copy hashes separately when metadata changed. POST_UPDATE_CHECKS.json reports the present update; earlier execution audits retain their original meaning.


AUTOMATIC TRAINING: TRAIN_CV.ps1 schedules all three seeds and five folds. Use -CheckOnly to validate without training. Details: TRAINING_README.txt. Existing-weight inference remains REPLAY.ps1.
