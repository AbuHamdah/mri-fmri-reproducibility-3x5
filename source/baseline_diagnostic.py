import json
from pathlib import Path
import numpy as np,tensorflow as tf
P=Path('/package');tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2);tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False)
d=P/'selected/mri_baseline/adopted_models/seed42_fold1';m=tf.keras.models.load_model(d/'model.keras',compile=False);c=json.loads((P/'selected/mri_baseline/calibration/seed42_fold1.json').read_text());ids=sorted({r['subject_id'] for r in json.loads((P/'INPUTS.json').read_text())});features=np.load(P/'selected/mri_baseline/cached_features.npy');head=m.layers[-1];encoder=tf.keras.Model(m.inputs,m.layers[-2].output);headall=head(features,training=False).numpy().ravel();records=[]
for row in c['predictions']:
 i=ids.index(row['subject_id']);x=np.load(P/'inputs/mri'/f"{row['subject_id']}.npy")[None];f=encoder(x,training=False).numpy();records.append(dict(subject_id=row['subject_id'],expected=row['probability'],raw_full_model=float(m(x,training=False).numpy().ravel()[0]),cached_head_batch42=float(headall[i]),cached_head_single=float(head(features[i:i+1],training=False).numpy().ravel()[0]),encoder_feature_difference=float(np.max(abs(f-features[i:i+1])))))
report=dict(thread_count=2,cached_features_shape=list(features.shape),head_type=head.__class__.__name__,records=records)
(P/'reproduction_run/BASELINE_PRECISION_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
