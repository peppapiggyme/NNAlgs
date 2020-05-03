from config.DatasetBuilder import DecayModePi0varBuilder
from nnalgs.base.GeneratorLoader import BaseGenLoader
from nnalgs.base.IDataset import Director


class DecayModePi0varTrainGenLoader(BaseGenLoader):
    """
    Gammatautau ntuple loading - pi0 BDT vars
    """

    def __init__(self):
        self._gens = tuple()
        for mode in ["Train", "Validation"]:
            dataset_builder = Director(DecayModePi0varBuilder())
            dataset_builder.build_dataset(mode=mode)
            dataset = dataset_builder.get_dataset()
            self._gens += (dataset.obj,)

        super().__init__()

    def generators(self):
        """
        :return: (TrainGen, ValidGen)
        """
        return self._gens
