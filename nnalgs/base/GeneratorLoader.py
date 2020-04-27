class BaseGenLoader(object):
    def __init__(self):
        pass

    @property
    def gen_obj(self):
        """
        :return: (TrainGen, ValidGen)
        """
        return None
