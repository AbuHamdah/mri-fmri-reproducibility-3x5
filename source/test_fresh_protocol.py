"""Validation and initialization tests; no optimizer updates or cohort training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from train_fresh import prepare, VERSION
from model_utils import label_targets, split_indices
from train_fusion_fresh import validate_branch


class ProtocolTests(unittest.TestCase):
    def test_all_splits(self):
        root=Path(__file__).resolve().parents[1]
        for seed in (42,43,44):
            for fold in range(1,6):
                ids,y,_,split,indices,_=prepare(root,seed,fold)
                self.assertEqual(len(ids),42)
                self.assertEqual(sum(len(v) for v in indices.values()),42)
                self.assertTrue(np.isin(y,[0,1]).all())

    def test_labels_unchanged(self):
        np.testing.assert_array_equal(label_targets(['b','a'],{'a':0,'b':1}),[1,0])
        with self.assertRaises(ValueError):label_targets(['a'],{'a':.8})

    def test_overlapping_split_rejected(self):
        with self.assertRaises(ValueError):
            split_indices(['a','b','c'],{'train':['a'],'validation':['b'],'test':['a','c']})

    def test_old_fusion_sources_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'protocol.json').write_text('{}')
            with self.assertRaises(ValueError):validate_branch(p,'mri',{})

    def test_fresh_architectures_read_config_only(self):
        import tensorflow as tf
        import zipfile
        from fresh_model import fresh_model
        root=Path(__file__).resolve().parents[1]
        _,_,_,_,_,architectures=prepare(root,42,1)
        original=zipfile.ZipFile.read
        reads=[]
        def guarded(archive,name,*args,**kwargs):
            self.assertEqual(name,'config.json');reads.append(name)
            return original(archive,name,*args,**kwargs)
        for path in architectures.values():
            tf.keras.backend.clear_session()
            with patch.object(zipfile.ZipFile,'read',guarded):model=fresh_model(path,42)
            self.assertTrue(model.trainable_weights)
            self.assertFalse(model.non_trainable_weights)
            self.assertEqual(model.output_shape[-1],1)
        self.assertEqual(len(reads),2)

    def test_fusion_normalization_ignores_heldout_rows(self):
        from fusion_utils import build_heads,validation_weight
        x=np.array([[.1,.2],[.8,.9],[.7,.3],[.2,.8]],dtype='float32')
        heads=build_heads(x,x,x,[0,1],42)
        changed=x.copy();changed[2:]=1000
        other=build_heads(changed,changed,changed,[0,1],42)
        for left,right in zip(heads,other):
            for a,b in zip(left.get_weights(),right.get_weights()):np.testing.assert_array_equal(a,b)
        y=np.array([0,1,1,0]);alpha=validation_weight(x,y,[0,1])
        y[2:]=1-y[2:]
        self.assertEqual(alpha,validation_weight(x,y,[0,1]))


if __name__=='__main__':unittest.main()
