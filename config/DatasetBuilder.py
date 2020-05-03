import numpy as np

from nnalgs.base.DatasetBuilder import AbsDatasetBuilder
from nnalgs.algs.DataGenerators import DecayModeDataGenerator
from nnalgs.utils.Common import walk_dir


class DecayModePi0varBuilder(AbsDatasetBuilder):
    def build_metadata(self):

        # Copy from self to Dataset object
        self._dataset.copyable = ['paths', 'tree_name', 'lmdb_dir', 'json_dir',
                                  'length', 'mode', 'lmdb_kwargs',
                                  'data', 'sel_vars', 'dtype', 'n_steps',
                                  'log_vars', 'logabs_vars', 'branch', 'shape']

        self._dataset.concrete_dataset = DecayModeDataGenerator
        self._dataset.paths = walk_dir("/data/zhangb/root_files_v61/", "data-tree")
        self._dataset.tree_name = "tree"
        self._dataset.lmdb_dir = "data/lmdb/decaymode/"
        self._dataset.json_dir = "data/json/decaymode/"
        self._dataset.length = 3e7
        self._dataset.mode = "Dummy"

        self._dataset.lmdb_kwargs = {
            "obj_name": "DecayModeDatasetCreator",
            "paths": self._dataset.paths,
            "tree_name": self._dataset.tree_name,
            "lmdb_dir": self._dataset.lmdb_dir,
            "json_dir": self._dataset.json_dir,
            "mode": self._dataset.mode,
        }

    def build_vars(self):
        # add new variables to branch here
        self._dataset.data = {
            "ChargedPFO": ["ChargedPFO.phi",
                           "ChargedPFO.dphi",
                           "ChargedPFO.eta",
                           "ChargedPFO.deta",
                           "ChargedPFO.pt",
                           "ChargedPFO.jetpt", ],
            "NeutralPFO": ["NeutralPFO.phi",
                           "NeutralPFO.dphi",
                           "NeutralPFO.eta",
                           "NeutralPFO.deta",
                           "NeutralPFO.pt",
                           "NeutralPFO.jetpt",
                           "NeutralPFO.FIRST_ETA",
                           "NeutralPFO.SECOND_R",
                           "NeutralPFO.DELTA_THETA",
                           "NeutralPFO.CENTER_LAMBDA",
                           "NeutralPFO.LONGITUDINAL",
                           "NeutralPFO.SECOND_ENG_DENS",
                           # "NeutralPFO.ENG_FRAC_EM",  #
                           "NeutralPFO.ENG_FRAC_CORE",
                           "NeutralPFO.NPosECells_EM1",
                           "NeutralPFO.NPosECells_EM2",
                           # "NeutralPFO.nHitsInEM1",  #
                           "NeutralPFO.ptSubRatio",
                           "NeutralPFO.energyfrac_EM2",
                           "NeutralPFO.EM1CoreFrac",
                           "NeutralPFO.secondEtaWRTClusterPosition_EM1",
                           "NeutralPFO.firstEtaWRTClusterPosition_EM1",
                           "NeutralPFO.secondEtaWRTClusterPosition_EM2",
                           ],
            "ShotPFO": ["ShotPFO.phi",
                        "ShotPFO.dphi",
                        "ShotPFO.eta",
                        "ShotPFO.deta",
                        "ShotPFO.pt",
                        "ShotPFO.jetpt", ],
            "ConvTrack": ["ConvTrack.phi",
                          "ConvTrack.dphi",
                          "ConvTrack.eta",
                          "ConvTrack.deta",
                          "ConvTrack.pt",
                          "ConvTrack.jetpt", ],
            "Label": ["TauJets.truthDecayMode"],
        }

        self._dataset.sel_vars = ["TauJets.pt", "TauJets.truthPtVis", "TauJets.eta", "TauJets.truthEtaVis",
                                  "TauJets.nTracks", "TauJets.truthProng", "TauJets.IsTruthMatched",
                                  "TauJets.truthDecayMode", "TauJets.mcEventNumber"]

        # don't modify this
        self._dataset.dtype = {
            "ChargedPFO": np.float32,
            "NeutralPFO": np.float32,
            "ShotPFO": np.float32,
            "ConvTrack": np.float32,
            "Label": np.long,
        }

        # number of objects per tau
        self._dataset.n_steps = {
            "ChargedPFO": 3,
            "NeutralPFO": 10,
            "ShotPFO": 6,
            "ConvTrack": 4,
            "Label": None,
        }

        self._dataset.log_vars = {
            "ChargedPFO.pt": 1e-3,
            "ChargedPFO.jetpt": 1e-3,
            "NeutralPFO.pt": 1e-3,
            "NeutralPFO.jetpt": 1e-3,
            "NeutralPFO.SECOND_R": 1e-3,
            "NeutralPFO.CENTER_LAMBDA": 1e-1,
            "NeutralPFO.SECOND_ENG_DENS": 1e-6,
            "NeutralPFO.secondEtaWRTClusterPosition_EM1": 1e-6,
            "NeutralPFO.secondEtaWRTClusterPosition_EM2": 1e-6,
            "ShotPFO.pt": 1e-3,
            "ShotPFO.jetpt": 1e-3,
            "ConvTrack.pt": 1e-3,
            "ConvTrack.jetpt": 1e-3,
        }

        self._dataset.logabs_vars = {
            "NeutralPFO.ptSubRatio": 1e-6,
        }

        self._dataset.branch = ["ChargedPFO", "NeutralPFO", "ShotPFO", "ConvTrack", "Label"]

        self._dataset.shape = dict()
        for name in self._dataset.branch:
            if name == "Label":
                self._dataset.shape[name] = ()  # HC ! Be careful !
            else:
                self._dataset.shape[name] = (self._dataset.n_steps[name], len(self._dataset.data[name]))


class DecayModePi0varTrainBuilder(DecayModePi0varBuilder):

    def build_metadata(self):
        super().build_metadata()
        self._dataset.mode = "Train"


class DecayModePi0varValidBuilder(DecayModePi0varBuilder):

    def build_metadata(self):
        super().build_metadata()
        self._dataset.mode = "Validation"
