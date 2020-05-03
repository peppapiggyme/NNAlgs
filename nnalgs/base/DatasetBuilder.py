import abc

from nnalgs.base.IDataset import IDataset


class AbsDatasetBuilder(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._dataset = None

    def get_dataset(self):
        return self._dataset

    def new_dataset(self):
        self._dataset = IDataset()

    @abc.abstractmethod
    def build_metadata(self, mode):
        """
        descriptions
        """

        raise NotImplementedError

    @abc.abstractmethod
    def build_vars(self):
        """
        descriptions
        """

        raise NotImplementedError
