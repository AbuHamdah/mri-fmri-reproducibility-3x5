"""Train new fusion heads only from verified fresh branch outputs."""
import argparse
from pathlib import Path
import numpy as np
from train_fresh import VERSION, prepare, read, write, sha


def validate_branch(folder, kind, split):
    protocol = read(folder / 'protocol.json')
    if (protocol.get('objective_version') != VERSION or
            protocol.get('initialization') != 'fresh_architecture_only' or
            protocol.get('modality') != kind or protocol.get('split') != split or
            protocol.get('model_sha256') != sha(folder / 'model.keras')):
        raise ValueError('Fusion requires matching, unchanged fresh branch outputs: ' + str(folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mri', type=Path, required=True)
    parser.add_argument('--fmri', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[42,43,44], default=42)
    parser.add_argument('--fold', type=int, choices=range(1,6), default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root, output = args.package.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(root) or args.epochs < 1:
        parser.error('Use positive epochs and a new output directory outside the package')
    ids, y, paths, split, indices, _ = prepare(root,args.seed,args.fold)
    folders = [args.mri.resolve(),args.fmri.resolve()]
    for folder,kind in zip(folders,['mri','fmri']):
        validate_branch(folder,kind,split)
    if args.check_only:
        print('Fresh branch identities and partitions verified');return
    import tensorflow as tf
    from fusion_utils import build_heads,validation_weight
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    models = [tf.keras.models.load_model(p/'model.keras',compile=False) for p in folders]
    features=[];probabilities=[]
    for kind,model in zip(['mri','fmri'],models):
        extractor=tf.keras.Model(model.input,[model.layers[-1].input,model.output])
        ff=[];pp=[]
        for sid in ids:
            data=np.concatenate([np.load(p,allow_pickle=False).reshape((-1,)+tuple(model.input_shape[1:])) for p in paths[kind][sid]])
            f,p=extractor(data,training=False);ff.append(f.numpy().mean(0));pp.append(float(tf.reduce_mean(p)))
        features.append(np.asarray(ff));probabilities.append(pp)
    pairs=np.asarray(probabilities,dtype='float32').T
    stack,feature=build_heads(pairs,*features,indices['train'],args.seed)
    output.mkdir(parents=True)
    write(output/'SPLIT.json',split)
    alpha=validation_weight(pairs,y,indices['validation'])
    def bce(target,p):
        p=np.clip(p,1e-7,1-1e-7);return float(np.mean(-target*np.log(p)-(1-target)*np.log1p(-p)))
    for name,model,inputs in [('stacking',stack,pairs),('feature',feature,features)]:
        optimizer=tf.keras.optimizers.Adam(1e-5);rng=np.random.default_rng(args.seed)
        val=indices['validation'];checkpoint=output/(name+'.weights.h5')
        best=bce(y[val],model(inputs,training=False).numpy().ravel()[val]);model.save_weights(checkpoint)
        history=[dict(epoch=0,validation_bce=best)]
        for epoch in range(1,args.epochs+1):
            for offset in rng.permutation(indices['train']):
                batch=[v[offset:offset+1] for v in inputs] if isinstance(inputs,list) else inputs[offset:offset+1]
                with tf.GradientTape() as tape:
                    p=model(batch,training=True)
                    loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.cast(y[offset:offset+1,None],p.dtype),p))
                    if model.losses:loss+=tf.add_n(model.losses)
                grads=tape.gradient(loss,model.trainable_variables)
                if any(g is None or not np.isfinite(g.numpy()).all() for g in grads):raise ValueError('Nonfinite fusion gradient')
                optimizer.apply_gradients(zip(grads,model.trainable_variables))
            score=bce(y[val],model(inputs,training=False).numpy().ravel()[val]);history.append(dict(epoch=epoch,validation_bce=score))
            if score<best:best=score;model.save_weights(checkpoint)
        model.load_weights(checkpoint);model.save(output/(name+'.keras'),save_format='keras_v3')
        pred=model(inputs,training=False).numpy().ravel()
        write(output/(name+'_history.json'),history)
        write(output/(name+'_test_predictions.json'),[dict(subject_id=ids[i],true_label=int(y[i]),probability=float(pred[i])) for i in indices['test']])
    weighted=alpha*pairs[:,0]+(1-alpha)*pairs[:,1]
    write(output/'weighted_test_predictions.json',[dict(subject_id=ids[i],true_label=int(y[i]),probability=float(weighted[i])) for i in indices['test']])
    write(output/'protocol.json',dict(objective_version=VERSION,initialization='fresh_heads',normalization='training_only',
          validation_alpha=alpha,split=split,controller='fixed',epochs=args.epochs,
          branches=[dict(path=str(p),sha256=sha(p/'model.keras')) for p in folders]))


if __name__=='__main__':main()
