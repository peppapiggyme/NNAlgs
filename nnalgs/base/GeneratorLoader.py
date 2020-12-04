class BaseGenLoader(object):
    def __init__(self):
        pass

    def generators(self):
        """
        :return: (TrainGen, ValidGen， TestGen)
        """
        return tuple((None, None, None))
