import lmdb
import numpy as np
from itertools import count
from keras.utils import to_categorical
from nnalgs.utils.Enum import N_CLASS_DECAYMODE


def decaymode_generator(lmdb_dir,
                        name,
                        shape,
                        dtype):
    env = lmdb.open(lmdb_dir, readonly=True, readahead=False, meminit=False)
    """
    https://lmdb.readthedocs.io/en/release/#lmdb.Environment
    """

    with env.begin(write=False) as txn:
        for i in count():
            array = txn.get("{}-{:09d}".format(name, i).encode())
            array = np.frombuffer(array, dtype=dtype).reshape(shape)

            if name == "Label":
                array = to_categorical(array, N_CLASS_DECAYMODE)

            yield array
