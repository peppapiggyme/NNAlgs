from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, TimeDistributed, Concatenate, Bidirectional
from keras.layers import Layer, Activation, BatchNormalization
from keras import backend as kbe

# for keras tuner
from tensorflow.keras import models as tfk_models
from tensorflow.keras import layers as tfk_layers

import yaml


# =============
# Custom Layers
# =============

class Sum(Layer):
    """Simple sum layer.

    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on
    their own

    See Dan's implementation:
    https://gitlab.cern.ch/deep-sets-example/higgs-regression-training/blob/master/SumLayer.py

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * kbe.cast(mask, kbe.dtype(x))[:, :, None]
        return kbe.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask=None):
        return None


class SumTFK(tfk_layers.Layer):
    """Simple sum layer.

    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on
    their own

    See Dan's implementation:
    https://gitlab.cern.ch/deep-sets-example/higgs-regression-training/blob/master/SumLayer.py

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * kbe.cast(mask, kbe.dtype(x))[:, :, None]
        return kbe.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask=None):
        return None


# =================
# Functional models
# =================

def ModelDSNN(config_file, mask_value=0.0):
    para = yaml.full_load(open(config_file))["DSNN"]
    bn = True if para["batch_norm"] == 1 else False

    # Branch 1
    x_1 = Input(shape=(para["n_steps"]["ChargedPFO"], para["n_features"]["ChargedPFO"]))
    b_1 = Masking(mask_value=mask_value)(x_1)
    for x in range(para["n_tdd"]["ChargedPFO"]):
        b_1 = TimeDistributed(Dense(para["n_inputs"]["ChargedPFO"][x]))(b_1)
        b_1 = Activation("relu")(b_1)
    b_1 = Sum()(b_1)
    for x in range(para["n_h"]["ChargedPFO"]):
        b_1 = Dense(para["n_hiddens"]["ChargedPFO"][x])(b_1)
        b_1 = Activation("relu")(b_1)
    if bn:
        b_1 = BatchNormalization()(b_1)

    # Branch 2
    x_2 = Input(shape=(para["n_steps"]["NeutralPFO"], para["n_features"]["NeutralPFO"]))
    b_2 = Masking(mask_value=mask_value)(x_2)
    for x in range(para["n_tdd"]["NeutralPFO"]):
        b_2 = TimeDistributed(Dense(para["n_inputs"]["NeutralPFO"][x]))(b_2)
        b_2 = Activation("relu")(b_2)
    b_2 = Sum()(b_2)
    for x in range(para["n_h"]["NeutralPFO"]):
        b_2 = Dense(para["n_hiddens"]["NeutralPFO"][x])(b_2)
        b_2 = Activation("relu")(b_2)
    if bn:
        b_2 = BatchNormalization()(b_2)

    # Branch 3
    x_3 = Input(shape=(para["n_steps"]["ShotPFO"], para["n_features"]["ShotPFO"]))
    b_3 = Masking(mask_value=mask_value)(x_3)
    for x in range(para["n_tdd"]["ShotPFO"]):
        b_3 = TimeDistributed(Dense(para["n_inputs"]["ShotPFO"][x]))(b_3)
        b_3 = Activation("relu")(b_3)
    b_3 = Sum()(b_3)
    for x in range(para["n_h"]["ShotPFO"]):
        b_3 = Dense(para["n_hiddens"]["ShotPFO"][x])(b_3)
        b_3 = Activation("relu")(b_3)
    if bn:
        b_3 = BatchNormalization()(b_3)

    # Branch 4
    x_4 = Input(shape=(para["n_steps"]["ConvTrack"], para["n_features"]["ConvTrack"]))
    b_4 = Masking(mask_value=mask_value)(x_4)
    for x in range(para["n_tdd"]["ConvTrack"]):
        b_4 = TimeDistributed(Dense(para["n_inputs"]["ConvTrack"][x]))(b_4)
        b_4 = Activation("relu")(b_4)
    b_4 = Sum()(b_4)
    for x in range(para["n_h"]["ConvTrack"]):
        b_4 = Dense(para["n_hiddens"]["ConvTrack"][x])(b_4)
        b_4 = Activation("relu")(b_4)
    if bn:
        b_4 = BatchNormalization()(b_4)

    # Merge
    merged = Concatenate()([b_1, b_2, b_3, b_4])
    merged = Dense(para["n_fc1"])(merged)
    merged = Activation("relu")(merged)
    merged = Dense(para["n_fc2"])(merged)
    merged = Activation("relu")(merged)

    y = Dense(para["n_classes"], activation="softmax")(merged)

    return Model(inputs=[x_1, x_2, x_3, x_4], outputs=y)


def ModelLSTM(config_file, mask_value=0.0, unroll=True):

    global lstm_1, lstm_2, lstm_3, lstm_4

    para = yaml.full_load(open(config_file))["LSTM"]

    go_backwards = True if para["backwards"] == 1 else False
    bi_lstm = True if para["bidirectional"] == 1 else False

    # Branch 1
    x_1 = Input(shape=(para["n_steps"]["ChargedPFO"], para["n_features"]["ChargedPFO"]))
    mask_1 = Masking(mask_value=mask_value)(x_1)
    lstm_1 = TimeDistributed(Dense(para["n_inputs"]["ChargedPFO"], activation="relu"))(mask_1)
    for i in range(para["n_layers"]["ChargedPFO"]):
        seq = True if i < (para["n_layers"]["ChargedPFO"] - 1) else False

        lstm_here = LSTM(para["n_hiddens"]["ChargedPFO"], return_sequences=seq,
                         unroll=unroll, go_backwards=go_backwards)
        if bi_lstm:
            lstm_1 = Bidirectional(lstm_here)(lstm_1)
        else:
            lstm_1 = lstm_here(lstm_1)

    # Branch 2
    x_2 = Input(shape=(para["n_steps"]["NeutralPFO"], para["n_features"]["NeutralPFO"]))
    mask_2 = Masking(mask_value=mask_value)(x_2)
    for i in range(para["n_tdd_layers"]["NeutralPFO"]):
        lstm_2 = TimeDistributed(Dense(para["n_inputs"]["NeutralPFO"][i], activation="relu"))(mask_2)
    for i in range(para["n_layers"]["NeutralPFO"]):
        seq = True if i < (para["n_layers"]["NeutralPFO"] - 1) else False

        lstm_here = LSTM(para["n_hiddens"]["NeutralPFO"], return_sequences=seq,
                         unroll=unroll, go_backwards=go_backwards)

        if bi_lstm:
            lstm_2 = Bidirectional(lstm_here)(lstm_2)
        else:
            lstm_2 = lstm_here(lstm_2)

    # Branch 3
    x_3 = Input(shape=(para["n_steps"]["ShotPFO"], para["n_features"]["ShotPFO"]))
    mask_3 = Masking(mask_value=mask_value)(x_3)
    lstm_3 = TimeDistributed(Dense(para["n_inputs"]["ShotPFO"], activation="relu"))(mask_3)
    for i in range(para["n_layers"]["ShotPFO"]):
        seq = True if i < (para["n_layers"]["ShotPFO"] - 1) else False

        lstm_here = LSTM(para["n_hiddens"]["ShotPFO"], return_sequences=seq,
                         unroll=unroll, go_backwards=go_backwards)
        if bi_lstm:
            lstm_3 = Bidirectional(lstm_here)(lstm_3)
        else:
            lstm_3 = lstm_here(lstm_3)

    # Branch 4
    x_4 = Input(shape=(para["n_steps"]["ConvTrack"], para["n_features"]["ConvTrack"]))
    mask_4 = Masking(mask_value=mask_value)(x_4)
    lstm_4 = TimeDistributed(Dense(para["n_inputs"]["ConvTrack"], activation="relu"))(mask_4)
    for i in range(para["n_layers"]["ConvTrack"]):
        seq = True if i < (para["n_layers"]["ConvTrack"] - 1) else False

        lstm_here = LSTM(para["n_hiddens"]["ConvTrack"], return_sequences=seq,
                         unroll=unroll, go_backwards=go_backwards)
        if bi_lstm:
            lstm_4 = Bidirectional(lstm_here)(lstm_4)
        else:
            lstm_4 = lstm_here(lstm_4)

    # Merge
    merged_branches = Concatenate()([lstm_1, lstm_2, lstm_3, lstm_4])

    dense_1 = Dense(para["n_fc1"], activation="relu")(merged_branches)
    dense_2 = Dense(para["n_fc2"], activation="relu")(dense_1)

    y = Dense(para["n_classes"], activation="softmax")(dense_2)

    return Model(inputs=[x_1, x_2, x_3, x_4], outputs=y)


# ====================================
# Functions for hyper-parameter tuning
# ====================================

def ModelBuildDSNN(hp):
    # ================
    # Hyper parameters
    # ================

    # use batch norm or not?
    # bn = hp.Choice('batch_norm', values=[True, False])
    bn = True

    # activation functions
    # act_ftn = hp.Choice('activation_function', values=["tanh", "relu"])
    act_ftn = "relu"

    # optimizers
    optimizer = hp.Choice('optimizer', values=["adam", "nadam"])

    # default TDD n layers and n nodes
    n_layers_tdd_default = hp.Int('n_layers_tdd_default', min_value=3, max_value=4, step=1)
    n_nodes_tdd_default = [
        hp.Int(f'n_nodes_tdd_default_{i}', min_value=10, max_value=40, step=3) for i in range(n_layers_tdd_default)
    ]

    # default FC n layers and n nodes
    n_layers_fc_default = hp.Int('n_layers_fc_default', min_value=3, max_value=5, step=1)
    n_nodes_fc_default = [
        hp.Int(f'n_nodes_fc_default_{i}', min_value=10, max_value=40, step=3) for i in range(n_layers_fc_default)
    ]

    # neutral TDD n layers and n nodes
    n_layers_tdd_neutral = hp.Int('n_layers_tdd_neutral', min_value=3, max_value=4, step=1)
    n_nodes_tdd_neutral = [
        hp.Int(f'n_nodes_tdd_neutral_{i}', min_value=40, max_value=90, step=5) for i in range(n_layers_tdd_neutral)
    ]

    # neutral FC n layers and n nodes
    n_layers_fc_neutral = hp.Int('n_layers_fc_neutral', min_value=3, max_value=5, step=1)
    n_nodes_fc_neutral = [
        hp.Int(f'_nodes_fc_neutral_{i}', min_value=30, max_value=80, step=4) for i in range(n_layers_fc_neutral)
    ]

    # Final layers
    n_nodes_final_1 = hp.Int('final_1', min_value=60, max_value=220, step=16)
    n_nodes_final_2 = hp.Int('final_2', min_value=20, max_value=100, step=8)

    # ============
    # Architecture
    # ============

    # Branch 1
    x_1 = tfk_layers.Input(shape=(3, 6))
    b_1 = tfk_layers.Masking(mask_value=0)(x_1)
    for x in range(n_layers_tdd_default):
        b_1 = tfk_layers.TimeDistributed(tfk_layers.Dense(units=n_nodes_tdd_default[x]))(b_1)
        b_1 = tfk_layers.Activation(act_ftn)(b_1)
    b_1 = SumTFK()(b_1)
    for x in range(n_layers_fc_default):
        b_1 = tfk_layers.Dense(units=n_nodes_fc_default[x])(b_1)
        b_1 = tfk_layers.Activation(act_ftn)(b_1)
    if bn:
        b_1 = tfk_layers.BatchNormalization()(b_1)

    # Branch 2
    x_2 = tfk_layers.Input(shape=(10, 21))
    b_2 = tfk_layers.Masking(mask_value=0)(x_2)
    for x in range(n_layers_tdd_neutral):
        b_2 = tfk_layers.TimeDistributed(tfk_layers.Dense(units=n_nodes_tdd_neutral[x]))(b_2)
        b_2 = tfk_layers.Activation(act_ftn)(b_2)
    b_2 = SumTFK()(b_2)
    for x in range(n_layers_fc_neutral):
        b_2 = tfk_layers.Dense(units=n_nodes_fc_neutral[x])(b_2)
        b_2 = tfk_layers.Activation(act_ftn)(b_2)
    if bn:
        b_2 = tfk_layers.BatchNormalization()(b_2)

    # Branch 3
    x_3 = tfk_layers.Input(shape=(6, 6))
    b_3 = tfk_layers.Masking(mask_value=0)(x_3)
    for x in range(n_layers_tdd_default):
        b_3 = tfk_layers.TimeDistributed(tfk_layers.Dense(units=n_nodes_tdd_default[x]))(b_3)
        b_3 = tfk_layers.Activation(act_ftn)(b_3)
    b_3 = SumTFK()(b_3)
    for x in range(n_layers_fc_default):
        b_3 = tfk_layers.Dense(units=n_nodes_fc_default[x])(b_3)
        b_3 = tfk_layers.Activation(act_ftn)(b_3)
    if bn:
        b_3 = tfk_layers.BatchNormalization()(b_3)

    # Branch 4
    x_4 = tfk_layers.Input(shape=(4, 6))
    b_4 = tfk_layers.Masking(mask_value=0)(x_4)
    for x in range(n_layers_tdd_default):
        b_4 = tfk_layers.TimeDistributed(tfk_layers.Dense(units=n_nodes_tdd_default[x]))(b_4)
        b_4 = tfk_layers.Activation(act_ftn)(b_4)
    b_4 = SumTFK()(b_4)
    for x in range(n_layers_fc_default):
        b_4 = tfk_layers.Dense(units=n_nodes_fc_default[x])(b_4)
        b_4 = tfk_layers.Activation(act_ftn)(b_4)
    if bn:
        b_4 = tfk_layers.BatchNormalization()(b_4)

    # Merge
    merged = tfk_layers.Concatenate()([b_1, b_2, b_3, b_4])
    merged = tfk_layers.Dense(units=n_nodes_final_1)(merged)
    merged = tfk_layers.Activation(act_ftn)(merged)
    merged = tfk_layers.Dense(units=n_nodes_final_2)(merged)
    merged = tfk_layers.Activation(act_ftn)(merged)

    y = tfk_layers.Dense(5, activation="softmax")(merged)

    model = tfk_models.Model(inputs=[x_1, x_2, x_3, x_4], outputs=y)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])

    return model


def ModelBuildLSTM(hp):
    global lstm_1, lstm_2, lstm_3, lstm_4

    # ================
    # Hyper parameters
    # ================

    # Do not use Bi-Directorial
    bi_lstm = False

    # use batch norm or not?
    bn = hp.Choice('batch_norm', values=[True, False])

    # activation functions
    # act_ftn = hp.Choice('activation_function', values=["tanh", "relu"])
    act_ftn = "relu"

    # optimizers
    optimizer = hp.Choice('optimizer', values=["adam", "nadam"])

    # default TDD n layers and n nodes
    n_layers_tdd_default = hp.Int('n_layers_tdd_default', min_value=1, max_value=2, step=1)
    n_nodes_tdd_default = [
        hp.Int(f'n_nodes_tdd_default_{i}', min_value=10, max_value=30, step=2) for i in range(n_layers_tdd_default)
    ]

    # default LSTM n layers and n nodes
    n_layers_lstm_default = 1
    n_nodes_lstm_default = hp.Int('n_nodes_lstm_default', min_value=10, max_value=30, step=2)

    # neutral TDD n layers and n nodes
    n_layers_tdd_neutral = hp.Int('n_layers_tdd_neutral', min_value=1, max_value=2, step=1)
    n_nodes_tdd_neutral = [
        hp.Int(f'n_nodes_tdd_neutral_{i}', min_value=30, max_value=80, step=5) for i in range(n_layers_tdd_neutral)
    ]

    # neutral LSTM n layers and n nodes
    n_layers_lstm_neutral = 1
    n_nodes_lstm_neutral = hp.Int('n_nodes_lstm_neutral', min_value=20, max_value=70, step=5)

    # Final layers
    n_nodes_final_1 = hp.Int('final_1', min_value=60, max_value=160, step=10)
    n_nodes_final_2 = hp.Int('final_2', min_value=20, max_value=100, step=8)

    # ============
    # Architecture
    # ============

    # Branch 1
    x_1 = tfk_layers.Input(shape=(3, 6))
    mask_1 = tfk_layers.Masking(mask_value=0.0)(x_1)
    for i in range(n_layers_tdd_default):
        lstm_1 = tfk_layers.TimeDistributed(tfk_layers.Dense(units=n_nodes_tdd_default[i], activation=act_ftn))(mask_1)
    for i in range(n_layers_lstm_default):
        seq = True if i < (n_layers_lstm_default - 1) else False
        lstm_ftn = tfk_layers.LSTM(n_nodes_lstm_default, return_sequences=seq, unroll=True, go_backwards=False)
        lstm_1 = tfk_layers.Bidirectional(lstm_ftn)(lstm_1) if bi_lstm else lstm_ftn(lstm_1)
    if bn:
        lstm_1 = tfk_layers.BatchNormalization()(lstm_1)

    # Branch 2
    x_2 = tfk_layers.Input(shape=(10, 21))
    mask_2 = tfk_layers.Masking(mask_value=0.0)(x_2)
    for i in range(n_layers_tdd_neutral):
        lstm_2 = tfk_layers.TimeDistributed(tfk_layers.Dense(units=n_nodes_tdd_neutral[i], activation=act_ftn))(mask_2)
    for i in range(n_layers_lstm_neutral):
        seq = True if i < (n_layers_lstm_neutral - 1) else False
        lstm_ftn = tfk_layers.LSTM(n_nodes_lstm_neutral, return_sequences=seq, unroll=True, go_backwards=False)
        lstm_2 = tfk_layers.Bidirectional(lstm_ftn)(lstm_2) if bi_lstm else lstm_ftn(lstm_2)
    if bn:
        lstm_2 = tfk_layers.BatchNormalization()(lstm_2)

    # Branch 3
    x_3 = tfk_layers.Input(shape=(6, 6))
    mask_3 = tfk_layers.Masking(mask_value=0.0)(x_3)
    for i in range(n_layers_tdd_default):
        lstm_3 = tfk_layers.TimeDistributed(tfk_layers.Dense(units=n_nodes_tdd_default[i], activation=act_ftn))(mask_3)
    for i in range(n_nodes_lstm_default):
        seq = True if i < (n_nodes_lstm_default - 1) else False
        lstm_ftn = tfk_layers.LSTM(n_nodes_lstm_default, return_sequences=seq, unroll=True, go_backwards=False)
        lstm_3 = tfk_layers.Bidirectional(lstm_ftn)(lstm_3) if bi_lstm else lstm_ftn(lstm_3)
    if bn:
        lstm_3 = tfk_layers.BatchNormalization()(lstm_3)

    # Branch 4
    x_4 = tfk_layers.Input(shape=(4, 6))
    mask_4 = tfk_layers.Masking(mask_value=0.0)(x_4)
    for i in range(n_layers_tdd_default):
        lstm_4 = tfk_layers.TimeDistributed(tfk_layers.Dense(units=n_nodes_tdd_default[i], activation=act_ftn))(mask_4)
    for i in range(n_nodes_lstm_default):
        seq = True if i < (n_nodes_lstm_default - 1) else False
        lstm_ftn = tfk_layers.LSTM(n_nodes_lstm_default, return_sequences=seq, unroll=True, go_backwards=False)
        lstm_4 = tfk_layers.Bidirectional(lstm_ftn)(lstm_4) if bi_lstm else lstm_ftn(lstm_4)
    if bn:
        lstm_4 = tfk_layers.BatchNormalization()(lstm_4)

    # Merge
    merged_branches = tfk_layers.Concatenate()([lstm_1, lstm_2, lstm_3, lstm_4])

    dense_1 = tfk_layers.Dense(units=n_nodes_final_1, activation=act_ftn)(merged_branches)
    dense_2 = tfk_layers.Dense(units=n_nodes_final_2, activation=act_ftn)(dense_1)

    y = tfk_layers.Dense(5, activation="softmax")(dense_2)

    model = tfk_models.Model(inputs=[x_1, x_2, x_3, x_4], outputs=y)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["categorical_accuracy"])

    return model
