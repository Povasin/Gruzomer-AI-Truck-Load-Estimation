"""Small reproducible image regression baseline; Python 3.10+, numpy and Pillow.

No pretrained weights, network access, filename features or test targets.
Validation is made from TRAIN groups only. Default method is selected there.
"""
from pathlib import Path
import argparse
import csv
import json
import random
import numpy as np
from PIL import Image, ImageOps
def read_table(path, fields):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=fields:
            raise ValueError(f'{path}: expected columns {fields}')
        rows=list(reader)
    if not rows or len({r['image_id'] for r in rows})!=len(rows):
        raise ValueError('Empty table or duplicate image IDs')
    if any(not r['image_id'] or Path(r['image_id']).name!=r['image_id'] or '/' in r['image_id'] or '\\' in r['image_id'] for r in rows):
        raise ValueError('Invalid image ID')
    return rows

def feature(path):
    with Image.open(path) as source:
        im=ImageOps.exif_transpose(source).convert('RGB')
        aspect=im.width/im.height
        small=np.asarray(im.resize((32,32),Image.Resampling.BILINEAR),dtype=np.float64)/255
        spatial=small.reshape(8,4,8,4,3).mean(axis=(1,3)).ravel()
        gray=small.mean(axis=2)
        texture=gray.reshape(4,8,4,8).std(axis=(1,3)).ravel()
        hist=np.concatenate([np.histogram(small[:,:,c],bins=8,range=(0,1))[0]/1024 for c in range(3)])
    return np.concatenate((spatial,texture,hist,[aspect]))

def features(image_dir,rows):
    result=[]
    for i,r in enumerate(rows):
        result.append(feature(Path(image_dir)/(r['image_id']+'.jpg')))
        if (i+1)%200==0: print(f'Features: {i+1}/{len(rows)}',flush=True)
    return np.asarray(result)

def fit_ridge(x,y,alpha):
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1
    z=(x-mean)/scale
    intercept=float(y.mean())
    weights=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-intercept))
    return dict(mean=mean,scale=scale,weights=weights,intercept=np.asarray(intercept))

def predict_ridge(model,x):
    return np.clip(((x-model['mean'])/model['scale'])@model['weights']+float(model['intercept']),0,100)

def metrics(y,p):
    errors=np.abs(y-p)
    return dict(mae=float(errors.mean()),within_10_pct=float(100*(errors<=10).mean()),count=len(y))

def train(args: argparse.Namespace) -> None:
    rows=read_table(args.train_csv,['image_id','load_pct'])
    groups=read_table(args.groups_csv,['image_id','group_id'])
    mapping={r['image_id']:r['group_id'] for r in groups}
    if set(mapping)!={r['image_id'] for r in rows}:
        raise ValueError('Group mapping must cover precisely train.csv')
    y=np.array([float(r['load_pct']) for r in rows])
    if not np.isfinite(y).all() or ((y<0)|(y>100)).any(): raise ValueError('Invalid train labels')
    x=features(args.images,rows)
    ordered=sorted(set(mapping.values()));random.Random(args.seed).shuffle(ordered)
    val_groups=set();number=0
    counts={g:sum(mapping[r['image_id']]==g for r in rows) for g in ordered}
    for group in ordered:
        if number>=round(len(rows)*.2): break
        val_groups.add(group);number+=counts[group]
    validation=np.array([mapping[r['image_id']] in val_groups for r in rows])
    if validation.all() or not validation.any(): raise ValueError('Not enough independent train groups')
    ridge=fit_ridge(x[~validation],y[~validation],args.alpha)
    ridge_metrics=metrics(y[validation],predict_ridge(ridge,x[validation]))
    median=float(np.median(y[~validation]))
    median_metrics=metrics(y[validation],np.full(validation.sum(),median))
    selected=args.method
    if selected=='auto': selected='ridge' if ridge_metrics['mae']<median_metrics['mae'] else 'median'
    final=fit_ridge(x,y,args.alpha)
    final.update(method=np.asarray(selected),median=np.asarray(float(np.median(y))))
    model_path=Path(args.model);model_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(model_path,**final)
    report=dict(seed=args.seed,alpha=args.alpha,features=x.shape[1],train_count=len(rows),
        fit_count=int((~validation).sum()),validation_count=int(validation.sum()),
        fit_group_count=len(ordered)-len(val_groups),validation_group_count=len(val_groups),
        median=median_metrics,ridge=ridge_metrics,selected_method=selected,
        note='Method selected on an internal grouped holdout of TRAIN; final model refitted on all TRAIN. No public/private targets used.')
    model_path.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

def predict(args):
    rows=read_table(args.test_csv,['image_id'])
    with np.load(args.model,allow_pickle=False) as saved:
        model={k:saved[k] for k in saved.files}
    if str(model['method'])=='median':
        values=np.full(len(rows),float(model['median']))
    else: values=predict_ridge(model,features(args.images,rows))
    if not np.isfinite(values).all(): raise ValueError('Nonfinite predictions')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','load_pct'])
        w.writerows((r['image_id'],f'{float(value):.6f}') for r,value in zip(rows,values))
    print(f'Saved {len(rows)} predictions.')

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--train-csv',required=True);t.add_argument('--groups-csv',required=True)
    t.add_argument('--images',required=True);t.add_argument('--model',default='model.npz')
    t.add_argument('--seed',type=int,default=20260915);t.add_argument('--alpha',type=float,default=100.0)
    t.add_argument('--method',choices=['auto','ridge','median'],default='auto');t.set_defaults(run=train)
    q=sub.add_parser('predict');q.add_argument('--model',default='model.npz');q.add_argument('--test-csv',required=True)
    q.add_argument('--images',required=True);q.add_argument('--output',default='submission.csv');q.set_defaults(run=predict)
    args=p.parse_args();args.run(args)

if __name__=='__main__': main()
