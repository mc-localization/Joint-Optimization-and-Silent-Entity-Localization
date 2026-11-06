"""Train/evaluate one neural configuration exported from an accepted paper run."""
from pathlib import Path
import argparse,copy,csv,json,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'])
    args=p.parse_args()
    if args.output.exists():p.error('--output must name a new directory')
    if not (args.data_dir/'independent_test.npz').is_file():p.error('Generate the full dataset first; see README.md')
    sys.path[:0]=[str(ROOT/'experiment'),str(ROOT/'experiment/runtime')]
    import numpy as np,torch
    import run_experiment_4rx as base
    import dnn10_width100_cnn50 as ten_layer
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    ten_layer.install_model(base)
    record=json.loads(args.config.read_text(encoding='utf-8'));cfg=copy.deepcopy(record['configuration'])
    cfg['paths'].update(data_dir=str(args.data_dir.resolve()),output_dir=str(args.output.resolve()),checkpoint_cache_dir=str(args.output.resolve()))
    cfg['training']['device']=args.device or cfg['training']['device']
    train,valid,_,_,_=base.load_splits(cfg,final=True,include_test=False)
    with np.load(args.data_dir/'independent_test.npz',allow_pickle=False) as z:
        test={k:z[k].copy() for k in ['X','x','y']}
    scaler=base.TargetScaler(train);length,start=record['length'],record['start']
    td,vd=[base.make_dataset(data,scaler,length,start,cfg) for data in [train,valid]]
    device=base.choose_device(cfg);history=[]
    base.TRAINING_OBSERVER=lambda epoch,tr,va:history.append(dict(epoch=epoch,train_mse_um2=tr,val_mse_um2=va))
    args.output.mkdir(parents=True)
    base.configure_seed(cfg,int(cfg['training']['seed']))
    state,loss,epoch,seconds=base.train_localizer(cfg,record['model'],length,td,vd,device,record['epochs'],record['patience'])
    model=base.build_localizer(cfg,record['model'],length).to(device);model.load_state_dict(state);model.eval()
    dataset=base.make_dataset(test,scaler,length,start,cfg);pred=[]
    with torch.no_grad():
        for xb,_ in base.make_loader(dataset,cfg,device,length,False):
            with torch.amp.autocast(device_type=device.type,enabled=bool(cfg['training'].get('amp',False)) and device.type=='cuda'):
                pred.append(base.forward_model(model,xb.to(device)).float().cpu().numpy())
    prediction=scaler.physical(np.concatenate(pred));truth=np.column_stack([test['x'],test['y']]);errors=np.linalg.norm(prediction-truth,axis=1)
    np.savez_compressed(args.output/'predictions.npz',predictions=prediction,truth=truth,errors_um=errors)
    torch.save(dict(state=state,val_loss=loss,best_epoch=epoch,seed=cfg['training']['seed']),args.output/'model.pt')
    result=dict(model=record['model'],length=length,start=start,mean_error_um=float(errors.mean()),median_error_um=float(np.median(errors)),accuracy_percent=100*float((errors<.8).mean()),training_seconds=seconds,
        seed=cfg['training']['seed'],max_epochs=record['epochs'],patience=record['patience'],epochs_completed=len(history),best_epoch=epoch,val_mse_um2=loss,early_stopped=len(history)<record['epochs'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    (args.output/'training_config.json').write_text(json.dumps(cfg,indent=2))
    with (args.output/'curve.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
