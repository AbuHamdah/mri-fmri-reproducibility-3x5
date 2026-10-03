raise SystemExit('Historical training entry point disabled. Use source/train_fresh.py or source/train_fusion_fresh.py with a new output directory.')
import json, time, hashlib, traceback, sys, urllib.request
from pathlib import Path
import numpy as np
import tensorflow as tf
from statistics_core import metrics, NAMES
R = Path('/experiment')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, allow_nan=False), encoding='utf-8')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def wh(m):
    return hashlib.sha256(b''.join((np.ascontiguousarray(w).tobytes() for w in m.get_weights()))).hexdigest()
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()

def evidence(y, p):
    a = dict(zip(NAMES, map(float, metrics(y, p))))
    return dict(n=len(y), balanced_accuracy=round(0.5 * (a['recall'] + a['specificity']), 6), brier=round(a['brier'], 6), accuracy=round(a['accuracy'], 6))

def decision(m, history, train_metrics, val_metrics, d, seed, fold, epoch):
    drop = m.get_layer('active_training_dropout')
    lr = float(tf.keras.backend.get_value(m.optimizer.learning_rate))
    before = dict(learning_rate=lr, dropout=float(drop.rate), optimizer_iterations=int(m.optimizer.iterations.numpy()))
    payload = dict(epoch=epoch, max_epochs=25, learning_rate=round(lr, 8), dropout=round(drop.rate, 2), recent_history=[{k: round(v, 6) for k, v in h.items()} for h in history[-3:]], training=train_metrics, validation=val_metrics, actions=['continue', 'reduce_lr', 'increase_dropout', 'stop'])
    request_body = dict(model='qwen2.5:3b', stream=False, format=dict(type='object', properties=dict(action=dict(type='string', enum=payload['actions'])), required=['action'], additionalProperties=False), options=dict(temperature=0, seed=seed, num_ctx=2048, num_predict=32, num_thread=4, num_gpu=0), messages=[dict(role='system', content='Control feature-fusion training using only supplied TRAIN and VALIDATION evidence. Continue if improving; reduce_lr halves learning rate; increase_dropout adds0.1 up to0.5; stop ends training now. No target test score. Choose one action. JSON action field only.'), dict(role='user', content=json.dumps(payload, separators=(',', ':')))])
    folder = d / 'agent_decisions' / f'epoch{epoch}'
    folder.mkdir(parents=True)
    for attempt in [1, 2]:
        a = folder / f'attempt{attempt}'
        a.mkdir()
        save(a / 'REQUEST.json', request_body)
        t = time.perf_counter()
        try:
            req = urllib.request.Request('http://host.docker.internal:11434/api/chat', data=json.dumps(request_body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = json.load(response)
            save(a / 'RAW_RESPONSE.json', raw)
            answer = json.loads(raw['message']['content'])
            assert answer['action'] in payload['actions']
            elapsed = time.perf_counter() - t
            break
        except Exception as ex:
            save(a / 'FAILURE.json', dict(error=repr(ex), seconds=time.perf_counter() - t))
            if attempt == 2:
                raise
    action = answer['action']
    if action == 'reduce_lr':
        m.optimizer.learning_rate.assign(lr * 0.5)
    elif action == 'increase_dropout':
        drop.rate = min(float(drop.rate) + 0.1, 0.5)
        m.train_function = None
        m.test_function = None
    after = dict(learning_rate=float(tf.keras.backend.get_value(m.optimizer.learning_rate)), dropout=float(drop.rate), optimizer_iterations=int(m.optimizer.iterations.numpy()))
    record = dict(seed=seed, fold=fold, epoch=epoch, action=action, before=before, after=after, successful_attempt=attempt, latency_seconds=elapsed, actual_live_response=True, request_sha256=sha(a / 'REQUEST.json'), response_sha256=sha(a / 'RAW_RESPONSE.json'), training_function_retrace_requested=action == 'increase_dropout')
    save(folder / 'ACTION.json', record)
    return record

def main():
    data = np.load(R / 'cache/FEATURES.npz')
    X = data['combined']
    y = data['labels'].astype(np.float32)
    ids = data['subject_ids'].tolist()
    plan = read(R / 'PLAN.json')
    assert sha(R / 'cache/FEATURES.npz') == plan['features_sha256']
    stage = sys.argv[1]
    index = []
    actions = []
    rows = []
    folds = []
    replays = []
    started = time.perf_counter()
    if stage == 'train':
        with urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=20) as response:
            save(R / 'OLLAMA_TAGS.json', json.load(response))
    else:
        assert len(read(R / 'ALL_TRAINING_MODELS_FROZEN.json')['records']) == 15
    for seed in [42, 43, 44]:
        for fold in range(1, 6):
            d = R / f'runs/seed{seed}_fold{fold}'
            split = read(d / 'SPLIT.json')
            sets = [set(split[k]) for k in ['train', 'validation', 'test']]
            assert set.union(*sets) == set(ids) and all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
            ix = {k: np.array([ids.index(s) for s in split[k]]) for k in ['train', 'validation', 'test']}
            tr = ix['train']
            va = ix['validation']
            pair = np.load(d / 'INPUT_PROBABILITIES.npz')
            assert pair['subject_ids'].tolist() == ids and np.array_equal(pair['labels'], y)
            prior = pair['pairs'].mean(1).astype(np.float32)
            p = np.clip(prior, 1e-07, 1 - 1e-07)
            base = (np.log(p) - np.log1p(-p))[:, None]
            if stage == 'train':
                sample = read(d / 'BOOTSTRAP_SAMPLE.json')
                boot = np.array([ids.index(s) for s in sample['sampled_child_ids']])
                assert set(boot).issubset(set(tr))
                mean = X[tr].mean(0)
                scale = np.maximum(X[tr].std(0), 0.01)
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed * 100 + fold)
                x = tf.keras.Input((256,), name='features')
                b = tf.keras.Input((1,), name='prior_logit')
                v = tf.keras.layers.Normalization(mean=mean, variance=scale ** 2)(x)
                v = tf.keras.layers.Dense(32, activation='relu')(v)
                v = tf.keras.layers.Dropout(0.2, name='active_training_dropout')(v)
                v = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform')(v)
                res = tf.keras.layers.Rescaling(0.75)(tf.keras.layers.Activation('tanh')(v))
                out = tf.keras.layers.Activation('sigmoid')(tf.keras.layers.Add()([b, res]))
                m = tf.keras.Model([x, b], out)
                m.add_loss(0.2 * tf.reduce_mean(tf.square(res)))
                m.compile(optimizer=tf.keras.optimizers.Adam(0.00075), loss='binary_crossentropy')
                initial = wh(m)
                assert initial == read(d / 'REFERENCE_TRAINING_EVIDENCE.json')['initial_weight_sha256']
                history = []
                run_actions = []
                t = time.perf_counter()
                for epoch in range(1, 26):
                    h = m.fit([X[boot], base[boot]], y[boot], validation_data=([X[va], base[va]], y[va]), epochs=1, batch_size=8, shuffle=True, verbose=0)
                    history.append(dict(epoch=epoch, loss=float(h.history['loss'][0]), val_loss=float(h.history['val_loss'][0]), learning_rate=float(tf.keras.backend.get_value(m.optimizer.learning_rate)), dropout=float(m.get_layer('active_training_dropout').rate), optimizer_iterations=int(m.optimizer.iterations.numpy())))
                    save(d / 'TRAINING_HISTORY.json', history)
                    if epoch in [10, 20]:
                        m.save(d / f'checkpoint_before_agent_epoch{epoch}.keras')
                        pt = m([X[tr], base[tr]], training=False).numpy().ravel()
                        pv = m([X[va], base[va]], training=False).numpy().ravel()
                        action = decision(m, history, evidence(y[tr], pt), evidence(y[va], pv), d, seed, fold, epoch)
                        run_actions.append(action)
                        actions.append(action)
                        print('Qwen training action', seed, fold, epoch, action['action'], flush=True)
                        if action['action'] == 'stop':
                            break
                m.save(d / 'model.keras')
                pva = m([X[va], base[va]], training=False).numpy().ravel()
                save(d / 'VALIDATION_PREDICTIONS.json', [dict(subject_id=ids[i], true_label=int(y[i]), probability=float(v)) for i, v in zip(va, pva)])
                ev = dict(seed=seed, fold=fold, epochs_trained=epoch, initial_weight_sha256=initial, final_weight_sha256=wh(m), gradient_training_seconds=time.perf_counter() - t - sum((a['latency_seconds'] for a in run_actions)), total_run_seconds=time.perf_counter() - t, current_training_children=split['train'], actual_fit_children=sample['unique_fit_children'], final_learning_rate=float(tf.keras.backend.get_value(m.optimizer.learning_rate)), final_dropout=float(m.get_layer('active_training_dropout').rate), optimizer_iterations=int(m.optimizer.iterations.numpy()), actions=run_actions)
                save(d / 'TRAINING_EVIDENCE.json', ev)
                index.append(dict(seed=seed, fold=fold, model_sha256=sha(d / 'model.keras'), split_sha256=sha(d / 'SPLIT.json'), history_sha256=sha(d / 'TRAINING_HISTORY.json'), evidence_sha256=sha(d / 'TRAINING_EVIDENCE.json')))
                save(R / 'PROGRESS.json', dict(stage='agent_controlled_training', completed=len(index), total=15, successful_decisions=len(actions)))
                print('Training complete', len(index), 'of15', flush=True)
            else:
                lock = next((r for r in read(R / 'ALL_TRAINING_MODELS_FROZEN.json')['records'] if (r['seed'], r['fold']) == (seed, fold)))
                assert sha(d / 'model.keras') == lock['model_sha256'] and sha(d / 'SPLIT.json') == lock['split_sha256']
                m = tf.keras.models.load_model(d / 'model.keras', compile=False)
                a = ix['test']
                pp = m([X[a], base[a]], training=False).numpy().ravel()
                again = tf.keras.models.load_model(d / 'model.keras', compile=False)
                pp2 = again([X[a], base[a]], training=False).numpy().ravel()
                delta = float(np.max(abs(pp - pp2)))
                assert delta < 1e-06
                replays.append(dict(seed=seed, fold=fold, reload_max_difference=delta))
                rr = [dict(seed=seed, test_fold=fold, subject_id=ids[i], true_label=int(y[i]), probability=float(v), prediction=int(v >= 0.5), threshold=0.5, model=f'runs/seed{seed}_fold{fold}/model.keras') for i, v in zip(a, pp)]
                save(d / 'TEST_PREDICTIONS.json', rr)
                rows.extend(rr)
                folds.append(dict(seed=seed, fold=fold, n=len(a), metrics=dict(zip(NAMES, map(float, metrics(y[a], pp))))))
    if stage == 'train':
        save(R / 'AGENT_DECISION_INDEX.json', actions)
        save(R / 'TRAINING_INDEX.json', [read(R / f"runs/seed{r['seed']}_fold{r['fold']}/TRAINING_EVIDENCE.json") for r in index])
        save(R / 'ALL_TRAINING_MODELS_FROZEN.json', dict(records=index, plan_sha256=sha(R / 'PLAN.json'), current_test_metrics_generated=False))
        save(R / 'PROGRESS.json', dict(stage='all_training_locked_pending_test', completed=15, successful_decisions=len(actions), seconds=time.perf_counter() - started))
    else:
        pooled = []
        for seed in [42, 43, 44]:
            rr = sorted([r for r in rows if r['seed'] == seed], key=lambda r: r['subject_id'])
            assert [r['subject_id'] for r in rr] == ids
            p = np.array([r['probability'] for r in rr])
            pooled.append(dict(seed=seed, n=42, metrics=dict(zip(NAMES, map(float, metrics(y, p))))))
        summary = dict(display_name=plan['display_name'], metrics={k: dict(mean=float(np.mean([r['metrics'][k] for r in pooled])), sd=float(np.std([r['metrics'][k] for r in pooled], ddof=1))) for k in NAMES}, successful_agent_decisions=len(read(R / 'AGENT_DECISION_INDEX.json')), agent_role='Actual control of learning rate, dropout and stopping during new gradient training')
        save(R / 'SUMMARY.json', summary)
        save(R / 'POOLED_PER_SEED.json', pooled)
        save(R / 'TEST_PREDICTIONS.json', rows)
        save(R / 'FOLD_METRICS.json', folds)
        save(R / 'NATIVE_REPLAY_AUDIT.json', dict(passed=True, models=15, rows=126, replays=replays))
        save(R / 'PROGRESS.json', dict(stage='test_complete_pending_host_audit', completed=15))
        print(json.dumps(summary), flush=True)
try:
    main()
except Exception:
    save(R / 'FAILURE.json', dict(stage=sys.argv[1], traceback=traceback.format_exc()))
    raise
