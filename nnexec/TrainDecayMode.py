"""
Training script
: using generator to feed data
"""

import os
from functools import partial

import numpy as np
import yaml
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.utils import multi_gpu_model

import config.GeneratorLoaders as GenModule
import nnalgs.algs.Models as ModelModule
from nnalgs.utils.CallBack import ParallelModelCheckpoint
from nnalgs.utils.Logger import get_logger

# fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def train(cfg):
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

    logger = get_logger('Train', 'INFO')

    # =====
    # Model
    # =====

    # build algs architecture, then print to console
    model_ftn = partial(getattr(ModelModule, cfg['model']), cfg['arch'])
    model = model_ftn()
    logger.info(model.summary())

    # ======
    # Saving
    # ======

    arch = model.to_json()
    with open(os.path.join(cfg['save_dir'], 'architecture.json'), 'w') as arch_file:
        arch_file.write(arch)

    # =========
    # Callbacks
    # =========

    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0001,
        patience=cfg['patience'], verbose=cfg['verbose'], restore_best_weights=True)

    model_checkpoint = ParallelModelCheckpoint(model,  # <- the original model
                                               path=os.path.join(cfg['save_dir'], 'weights-{epoch:02d}.h5'),
                                               monitor="val_loss", save_best_only=False, save_weights_only=True,
                                               verbose=cfg['verbose'])

    csv_logger = CSVLogger(os.path.join(cfg['save_dir'], 'log.csv'))

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=4e-5)

    callbacks = [early_stopping, model_checkpoint, csv_logger, reduce_lr]

    # =========
    # Generator
    # =========

    gen = getattr(GenModule, cfg['gen'])
    training_generator, validation_generator = gen().generators()

    # ========
    # Training
    # ========

    # the parallel model
    # model_gpu = multi_gpu_model(model, gpus=cfg['gpus'])
    model.compile(loss="categorical_crossentropy", optimizer=cfg["opt"], metrics=["categorical_accuracy"])

    history = model.fit(training_generator, epochs=cfg['epochs'], verbose=cfg['verbose'], callbacks=callbacks,
                        validation_data=validation_generator,
                        max_queue_size=4, workers=cfg['workers'], use_multiprocessing=False)

    val_loss, epoch = min(zip(history.history["val_loss"], history.epoch))
    val_acc,  epoch = max(zip(history.history["val_categorical_accuracy"], history.epoch))
    logger.info(f"\nMinimum val_loss {val_loss} at epoch {epoch + 1}\n")
    logger.info(f"\nMaximum val_categorical_accuracy {val_acc} at epoch {epoch + 1}\n")
