import unittest
from training_control import ValidationControl

class ControlTests(unittest.TestCase):
    def test_plateau_reduces_lr_before_stopping(self):
        c=ValidationControl(patience=3,lr_patience=2)
        self.assertEqual(c.observe(.4,.001),(True,.001,False))
        self.assertEqual(c.observe(.4,.001),(False,.001,False))
        self.assertEqual(c.observe(.5,.001),(False,.0005,False))
        self.assertEqual(c.observe(.5,.0005),(False,.0005,True))
    def test_improvement_resets_wait(self):
        c=ValidationControl(patience=2)
        c.observe(.5,.001);c.observe(.6,.001)
        self.assertEqual(c.observe(.4,.001),(True,.001,False))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):ValidationControl().observe(float('nan'),.001)

class BranchSmokeTests(unittest.TestCase):
    def test_real_updates_and_checkpoint_reload(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        import tensorflow as tf
        from train_fresh import branch,read
        tf.keras.utils.set_random_seed(42)
        def tiny(*args):
            return tf.keras.Sequential([tf.keras.layers.Input((2,)),tf.keras.layers.Dense(4,activation='relu'),tf.keras.layers.Dropout(.2),tf.keras.layers.Dense(1,activation='sigmoid')])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'output';out.mkdir()
            architecture=root/'architecture';architecture.write_text('test architecture')
            ids=['a','b','c','d','e','f'];y=np.array([0,1,0,1,0,1],dtype=float)
            paths={'mri':{}}
            for i,sid in enumerate(ids):
                p=root/(sid+'.npy');np.save(p,np.array([[i/6,1-i/6]],dtype='float32'));paths['mri'][sid]=[p]
            split={'train':ids[:2],'validation':ids[2:4],'test':ids[4:]}
            indices={k:np.array(v) for k,v in {'train':[0,1],'validation':[2,3],'test':[4,5]}.items()}
            with patch('fresh_model.fresh_model',tiny):
                branch(root,out,'mri',42,2,1,.001,(ids,y,paths,split,indices,{'mri':architecture}))
            history=read(out/'history.json');self.assertEqual(len(history),3)
            self.assertTrue(all('learning_rate' in r and 'training_bce' in r and 'dropout' in r for r in history))
            self.assertEqual([p['subject_id'] for p in read(out/'test_predictions.json')],ids[4:])
            self.assertLessEqual(read(out/'protocol.json')['max_reload_difference'],1e-6)

if __name__=='__main__':unittest.main()
