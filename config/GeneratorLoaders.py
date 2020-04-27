from nnalgs.base.BuilderDirector import Director
from nnalgs.base.GeneratorLoader import BaseGenLoader
from config.DatasetBuilder import \
    (DecayModePi0varTrainBuilder, DecayModePi0varValidBuilder)


class DecayModePi0varTrainGenLoader(BaseGenLoader):
    """
    Gammatautau ntuple loading - pi0 BDT vars
    """

    def __init__(self):
        self.gens = tuple()
        for builder in (DecayModePi0varTrainBuilder, DecayModePi0varValidBuilder):
            dataset_builder = Director(builder())
            dataset_builder.build_dataset()
            dataset = dataset_builder.get_dataset()
            self.gens += (dataset.obj,)

        super().__init__()

    @property
    def gen_obj(self):
        """
        :return: (TrainGen, ValidGen)
        """
        return self.gens
