#!/usr/bin/env python3
"""Run multiple depths independently, logging and exporting each."""
import argparse
import csv
import os
from pathlib import Path
import pickle
import subprocess
import sys
from datetime import datetime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', type=int, nargs='+', default=[2, 4, 6, 8])
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--script', default='cora_depth_experiment.py')
    p.add_argument('--outdir', default='depth_runs')
    args = p.parse_args()
    script = Path(args.script).resolve()
    if not script.exists():
        p.error(f'Cannot find experiment script: {script}')
    root = Path(args.outdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary = root / 'summary.csv'
    cols = ['layers','seed','model','final_test_acc','best_val_acc','best_val_acc_epoch','pkl','status']
    existing = set()
    if summary.exists():
        with summary.open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f): existing.add((r['layers'],r['seed'],r['model']))
    with summary.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if f.tell() == 0: writer.writeheader()
        for depth in args.layers:
            if depth < 2: p.error('layers must be >= 2')
            for seed in args.seeds:
                workdir = root / f'layers_{depth}' / f'seed_{seed}'
                workdir.mkdir(parents=True, exist_ok=True)
                log = workdir / 'train.log'
                env = os.environ.copy()
                env.update({'GCN_LAYERS':str(depth),'GCN_SEED':str(seed),'MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1'})
                # local Cora data and metric helper must be accessible from working dir
                # run from source directory; unique result file via per-experiment relocation
                print(f'Running layers={depth}, seed={seed} -> {log}',flush=True)
                old_files = set((script.parent/'results_pkls').glob(f'overall_comparison_cora_layers{depth}_*.pkl')) if (script.parent/'results_pkls').exists() else set()
                with log.open('w', encoding='utf-8') as lf:
                    cp = subprocess.run([sys.executable,'-u',str(script)], cwd=script.parent, env=env, stdout=lf, stderr=subprocess.STDOUT)
                all_files = set((script.parent/'results_pkls').glob(f'overall_comparison_cora_layers{depth}_*.pkl'))
                created = sorted(all_files-old_files, key=lambda x:x.stat().st_mtime)
                if cp.returncode or not created:
                    print(f'FAILED layers={depth} seed={seed} exit={cp.returncode}. See {log}',flush=True)
                    writer.writerow(dict(layers=depth,seed=seed,model='',final_test_acc='',best_val_acc='',best_val_acc_epoch='',pkl='',status='failed'))
                    f.flush()
                    continue
                src=created[-1]; dst=workdir/'metrics.pkl';src.replace(dst)
                with dst.open('rb') as pf: data=pickle.load(pf)
                for r in data['results']:
                    writer.writerow(dict(layers=depth,seed=seed,model=r['name'],final_test_acc=r['final_test_acc'],best_val_acc=r['best_val_acc'],best_val_acc_epoch=r['best_val_acc_epoch'],pkl=str(dst),status='ok'))
                f.flush()
                print(f'OK layers={depth} seed={seed}, metrics={dst}',flush=True)
    print(f'Finished. Summary: {summary}',flush=True)

if __name__=='__main__': main()
