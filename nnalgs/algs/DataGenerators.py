import os
from contextlib import closing

import lmdb
import numpy as np
from keras.utils import to_categorical

from nnalgs.algs.LMDBCreators import DecayModeLMDBCreator
from nnalgs.base.DataGenerator import BaseDataGenerator


class DecayModeDataGenerator(BaseDataGenerator):
    """Decay mode keras data generator"""

    def __init__(self, length, mode):
        """
        Docstring ...
        :param length:
        :param mode: to avoid name collision with the mode in dataset builder
        """
        ## HC length!
        self.length = int(length*0.8) if mode == "Train" else int(length*0.2)
        self.mode = mode
        self.list_idx = [n for n in range(self.length)]

        super().__init__()

    def _generate_input(self, list_idx_temp):
        """DocString"""
        input_features = []

        for name in self.branch[:-1]:
            assert name != "Label"

            # Initialization
            input_feature = np.empty((self.batch_size, *self.shape[name]))

            # Generate data
            for i, ID in enumerate(list_idx_temp):
                # Store sample
                arr_bin = self._txn.get("{}-{:09d}".format(name, ID).encode())
                arr_buf = np.frombuffer(arr_bin, dtype=self.dtype[name]).reshape(self.shape[name])
                input_feature[i, ...] = arr_buf

            input_features.append(input_feature)

        return input_features

    def _generate_target(self, list_idx_temp):
        """DocString"""
        name = self.branch[-1]
        assert name == "Label"

        y = np.empty((self.batch_size, *self.shape[name]), dtype=int)

        # Generate data
        for i, ID in enumerate(list_idx_temp):
            # Store sample
            arr_bin = self._txn.get("{}-{:09d}".format(name, ID).encode())
            arr_buf = np.frombuffer(arr_bin, dtype=self.dtype[name]).reshape(self.shape[name])
            y[i, ...] = arr_buf

        return to_categorical(y, self.n_classes)

    def load_lmdb(self, **kwargs):
        """
        load lmdb dataset
        :param kwargs: see `DatasetBuilder`
        :return:
        """
        
        self.store_dir = os.path.join(self.lmdb_dir, self.mode)
        if not os.path.exists(self.store_dir):
            print(f"Creating database in {self.store_dir} ...")
            with closing(
                    DecayModeLMDBCreator(
                        self.sel_vars, self.data, self.n_steps, self.dtype,
                        self.log_vars, self.logabs_vars, **kwargs
                    )
            ) as lmdb_creator:
                lmdb_creator.execute()
        else:
            print(f"Database already exists in {self.store_dir}, will not recreate ...")

        # reference: https://lmdb.readthedocs.io/en/release/#lmdb.Environment
        self._env = lmdb.open(self.store_dir, readonly=True, readahead=False, meminit=False)
        self._txn = self._env.begin(write=False)

        return None

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indices = np.arange(len(self.list_idx))
        if self.shuffle and self.mode == "Train":
            np.random.shuffle(self.indices)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_idx) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_idx_temp = [self.list_idx[k] for k in indices]

        # Generate data
        input_features = self._generate_input(list_idx_temp)

        if self.to_fit:
            target = self._generate_target(list_idx_temp)
            return input_features, target
        else:
            return input_features
