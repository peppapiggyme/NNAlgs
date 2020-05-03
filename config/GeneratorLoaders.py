from nnalgs.base.IDataset import Director
from nnalgs.base.GeneratorLoader import BaseGenLoader
from config.DatasetBuilder import \
    (DecayModePi0varTrainBuilder, DecayModePi0varValidBuilder)


class DecayModePi0varTrainGenLoader(BaseGenLoader):
    """
    Gammatautau ntuple loading - pi0 BDT vars
    """

    def __init__(self):
        self._gens = tuple()
        for builder in (DecayModePi0varTrainBuilder, DecayModePi0varValidBuilder):
            dataset_builder = Director(builder())
            dataset_builder.build_dataset()
            dataset = dataset_builder.get_dataset()
            self._gens += (dataset.obj,)

        super().__init__()

    def generators(self):
        """
        :return: (TrainGen, ValidGen)
        """
        return self._gens
