class BaseGenLoader(object):
    def __init__(self):
        pass

    def generators(self):
        """
        :return: (TrainGen, ValidGen)
        """
        return tuple((None, None))
