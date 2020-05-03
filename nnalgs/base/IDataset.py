class IDataset(object):

    def print_info(self, debug_me):
        if debug_me:
            from pprint import pprint as pp
            pp(self.__dict__)

    @property
    def obj(self):

        dataset = self.concrete_dataset(self.length, self.mode)

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
