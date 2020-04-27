class Director(object):

    def __init__(self, builder):
        self._builder = builder

    def build_dataset(self):
        self._builder.new_dataset()
        self._builder.build_metadata()
        self._builder.build_vars()

    def get_dataset(self):
        return self._builder.get_dataset()
