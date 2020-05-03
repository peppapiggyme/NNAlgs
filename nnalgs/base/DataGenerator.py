from abc import abstractmethod

from keras.utils import Sequence


class BaseDataGenerator(Sequence):
    """Keras data generator"""

    def __init__(self, batch_size=200, to_fit=True, n_classes=5,
                 shuffle=True):
        """
        Documentation
        :param batch_size:
        :param to_fit:
        :param n_classes:
        :param shuffle:
        """

        self.batch_size = batch_size
        self.to_fit = to_fit
        self.n_classes = n_classes
        self.shuffle = shuffle

        # LMDB objects
        self._env = None
        self._txn = None

        self.indices = None
        self.on_epoch_end()  # -> inherited from Sequence ...

    @abstractmethod
    def load_lmdb(self):
        """
        load lmdb dataset
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Denotes the number of batches per epoch"""
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """Generate one batch of data"""
        raise NotImplementedError

    @abstractmethod
    def _generate_input(self, list_idx_temp):
        """Generates data containing batch_size inputs"""
        raise NotImplementedError

    @abstractmethod
    def _generate_target(self, list_idx_temp):
        """Generates data containing batch_size target"""
        raise NotImplementedError
