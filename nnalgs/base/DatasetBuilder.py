import abc


class IDataset(object):

    def print_info(self, debug_me):
        if debug_me:
            from pprint import pprint as pp
            pp(self.__dict__)

    @property
    def obj(self):

        dataset = self.concrete_dataset(self.length, self.mode, 
                                        self.split, self.batch_size)

        for attr in self.copyable:
            setattr(dataset, attr, getattr(self, attr))

        dataset.load_lmdb(**self.lmdb_kwargs)

        return dataset


class Director(object):

    def __init__(self, builder):
        self._builder = builder

    def build_dataset(self, mode):
        self._builder.new_dataset()
        self._builder.build_metadata(mode)
        self._builder.build_vars()

    def get_dataset(self):
        return self._builder.get_dataset()


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
