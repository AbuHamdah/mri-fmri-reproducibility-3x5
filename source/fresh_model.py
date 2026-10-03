"""Construct a trainable network from architecture, without reading saved weights."""
import json
import zipfile
import tensorflow as tf


def fresh_model(path, seed):
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read('config.json'))

    allowed = {'Functional', 'Sequential', 'InputLayer', 'Conv3D', 'MaxPooling3D',
               'Flatten', 'Dense', 'Dropout', 'TimeDistributed', 'Bidirectional',
               'LSTM', 'Conv2D', 'MaxPooling2D'}

    def clean(node):
        if isinstance(node, list):
            for item in node:
                clean(item)
        elif isinstance(node, dict):
            if 'class_name' in node and isinstance(node.get('config'), dict):
                name = node['class_name']
                cfg = node['config']
                if 'name' in cfg:
                    if name not in allowed:
                        raise ValueError('Architecture requires review before fresh training: ' + name)
                    if name != 'InputLayer':
                        cfg['trainable'] = True
                    for key in list(cfg):
                        if key.endswith('_initializer'):
                            initializer = ('Orthogonal' if key == 'recurrent_initializer' else
                                           'Zeros' if key == 'bias_initializer' else 'GlorotUniform')
                            cfg[key] = {'class_name': initializer, 'config': {}}
            for value in node.values():
                clean(value)
    config.pop('compile_config', None)
    clean(config)
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.models.model_from_json(json.dumps(config))
    model.trainable = True
    return model
