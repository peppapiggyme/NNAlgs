"""
Tuning scripts
"""

import os
from functools import partial
from itertools import islice

import numpy as np
import tensorflow as tf
import yaml
from kerastuner import HyperParameters
from kerastuner.tuners import RandomSearch

import nnalgs.algs.Models as ModelModule
from nnalgs.utils.LMDBGen import decaymode_generator
from nnalgs.utils.Logger import get_logger

# fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def tune(cfg):
    # =========
    # Configure
    # =========

    cfg = yaml.full_load(open(cfg))
    # Go deep
    algName = [nm for nm in cfg][0]
    cfg = cfg[algName]

    # ======
    # Logger
    # ======

    logger = get_logger('Tune', 'INFO')

    # =======
    # Dataset
    # =======

    lmdb_dir = cfg['lmdb_dir']
    length = 4000
    train = 2000
    split = length - train

    s = np.arange(0, length)
    np.random.shuffle(s)

    y = list(islice(decaymode_generator(lmdb_dir, "Label", (), np.long), length))
    X_1 = list(islice(decaymode_generator(lmdb_dir, "ChargedPFO", (3, 6), np.float32), length))
    X_2 = list(islice(decaymode_generator(lmdb_dir, "NeutralPFO", (10, 21), np.float32), length))
    X_3 = list(islice(decaymode_generator(lmdb_dir, "ShotPFO", (6, 6), np.float32), length))
    X_4 = list(islice(decaymode_generator(lmdb_dir, "ConvTrack", (4, 6), np.float32), length))

    y = np.asarray(y)[s]
    X_1, X_2, X_3, X_4 = np.asarray(X_1)[s], np.asarray(X_2)[s], np.asarray(X_3)[s], np.asarray(X_4)[s]

    y_train = y[:-split]
    X_train_1, X_train_2, X_train_3, X_train_4 = X_1[:-split], X_2[:-split], X_3[:-split], X_4[:-split]

    y_valid = y[:-split]
    X_valid_1, X_valid_2, X_valid_3, X_valid_4 = X_1[-split:], X_2[-split:], X_3[-split:], X_4[-split:]

    # =====
    # Model
    # =====

    # build algs architecture, then print to console
    model_ftn = partial(getattr(ModelModule, cfg['model']), cfg['arch'])
    model = model_ftn()
    logger.info(model.summary())

    hp = HyperParameters()

    hp.Fixed("n_layers_tdd_default", 3)
    hp.Fixed("n_layers_fc_default", 3)

    tuner = RandomSearch(
        getattr(ModelModule, cfg['tune_model']),
        hyperparameters=hp,
        tune_new_entries=True,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=2,
        directory=os.path.join(cfg['save_dir'], cfg['tune']),
        project_name=cfg['tune'],
        distribution_strategy=tf.distribute.MirroredStrategy(),
    )

    logger.info('Search space summary: ')
    tuner.search_space_summary()

    logger.info('Now searching ... ')
    tuner.search([X_train_1, X_train_2, X_train_3, X_train_4], y_train,
                 steps_per_epoch=int(train / 200),
                 epochs=20,
                 validation_steps=int(split / 200),
                 validation_data=([X_valid_1, X_valid_2, X_valid_3, X_valid_4], y_valid),
                 workers=10,
                 verbose=0)

    logger.info('Done! ')
    models = tuner.get_best_models(num_models=8)
    tuner.results_summary()

    logger.info('Saving best models ... ')
    for i, model in enumerate(models):
        arch = model.to_json()
        with open(os.path.join(cfg['save_dir'], cfg['tune'], f'architecture-{i}.json'),
                  'w') as arch_file:
            arch_file.write(arch)
        model.save_weights(os.path.join(cfg['save_dir'], cfg['tune'], f'weights-{i}.h5'), 'w')
    logger.info('Done! ')
