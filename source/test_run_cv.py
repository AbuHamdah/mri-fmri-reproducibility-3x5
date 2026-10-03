"""Scheduler and aggregation tests using synthetic predictions, without training."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from train_fresh import write, read
from run_cv import jobs, preflight, aggregate, execute, SEEDS, FOLDS


class CVTests(unittest.TestCase):
    def test_job_counts_and_dependency_order(self):
        root=Path('/package');out=Path('/new')
        for kind in ('mri','fmri'):
            plan=jobs(root,out,kind,5,1,1e-5)
            self.assertEqual(len(plan),15)
            self.assertEqual({(j['seed'],j['fold']) for j in plan},{(s,f) for s in SEEDS for f in FOLDS})
        plan=jobs(root,out,'all',5,1,1e-5)
        self.assertEqual(len(plan),45)
        for i in range(0,45,3):
            self.assertEqual([j['kind'] for j in plan[i:i+3]],['mri','fmri','fusion'])
            self.assertEqual(len({(j['seed'],j['fold']) for j in plan[i:i+3]}),1)

    def test_real_five_fold_coverage(self):
        labels,parts=preflight(Path(__file__).resolve().parents[1])
        self.assertEqual(len(labels),42);self.assertEqual(len(parts),15)

    def test_duplicate_test_subject_rejected(self):
        def duplicate(*args):
            return ['a','b'],np.array([0,1]),{},dict(train=['b'],validation=[],test=['a']),{},{}
        with patch('run_cv.prepare',duplicate),self.assertRaises(ValueError):preflight(Path('/fake'))

    def test_failure_stops_dependent_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);plan=jobs(Path('/package'),out,'all',5,1,1e-5);calls=[]
            def fail(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=7)
            with self.assertRaises(RuntimeError):execute(plan,out,fail)
            self.assertEqual(len(calls),1);self.assertEqual(read(out/'FAILED.json')['exit_code'],7)
            self.assertFalse((out/'COMPLETE.json').exists())

    def test_pooled_metrics_and_duplicate_prediction_rejection(self):
        labels={str(i):i%2 for i in range(10)};parts={}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            for seed in SEEDS:
                for fold in FOLDS:
                    test=[str(2*(fold-1)),str(2*(fold-1)+1)]
                    split={'test':test,'train':[s for s in labels if s not in test],'validation':[]};parts[seed,fold]=split
                    folder=out/'mri'/f'seed{seed}_fold{fold}';folder.mkdir(parents=True)
                    write(folder/'SPLIT.json',split)
                    write(folder/'test_predictions.json',[dict(subject_id=s,true_label=labels[s],probability=.9 if labels[s] else .1) for s in test])
            aggregate(out,'mri',labels,parts)
            result=read(out/'SUMMARY.json')['results'][0]['metrics']['accuracy']
            self.assertEqual(result,{'mean':1.0,'sd':0.0})
            pooled=read(out/'POOLED_PER_SEED.json');self.assertEqual(len(pooled),3)
            self.assertTrue(all(p['subjects']==10 for p in pooled))
            path=out/'mri/seed42_fold1/test_predictions.json';rows=read(path);write(path,rows+rows[:1])
            with self.assertRaises(ValueError):aggregate(out,'mri',labels,parts)


if __name__=='__main__':unittest.main()
