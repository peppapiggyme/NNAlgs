class IDataset(object):

    def print_info(self, debug_me):
        if debug_me:
            from pprint import pprint as pp
            print("**** This the interface of the dataset builder ****")
            pp(self.__dict__)
            print("**** This the interface of the dataset builder ****")

    @property
    def obj(self):

        dataset = self.concrete_dataset(self.length, self.mode)

        for attr in self.copyable:
            setattr(dataset, attr, getattr(self, attr))

        dataset.load_lmdb(**self.lmdb_kwargs)

        return dataset
