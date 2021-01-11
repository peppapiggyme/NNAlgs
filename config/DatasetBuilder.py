import numpy as np

from nnalgs.algs.DataGenerators import DecayModeDataGenerator
from nnalgs.base.DatasetBuilder import AbsDatasetBuilder
from nnalgs.utils.Common import walk_dir


class DecayModePi0varBuilder(AbsDatasetBuilder):
    def build_metadata(self, mode):

        # Copy from self to Dataset object
        self._dataset.copyable = ['paths', 'tree_name', 'lmdb_dir', 'json_dir', 'length', 
                                  'mode', 'split', 'lmdb_kwargs', 'data', 'sel_vars', 'dtype', 
                                  'n_steps', 'log_vars', 'atan_vars', 'branch', 'shape']

        self._dataset.concrete_dataset = DecayModeDataGenerator
        self._dataset.paths = walk_dir("/data1/bowenzhang/r22-00/", "tree")
        self._dataset.tree_name = "tree"
        self._dataset.lmdb_dir = "NNAlgs/data/lmdb/decaymode/"
        self._dataset.json_dir = "NNAlgs/data/json/decaymode/"
        self._dataset.length = 13510000
        # this must be consistent with the LMDB creation !
        # see: nnalgs/algs/LMDBCreators -> _get_removed_indices
        self._dataset.split = {"Train": 0.6, "Validation": 0.2, "Test": 0.2}
        self._dataset.batch_size = {"Train": 200, "Validation": 500000, "Test": 500000}
        self._dataset.mode = mode

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
            "TauTrack": ["TauTrack.dphiECal",
                         "TauTrack.dphi",
                         "TauTrack.detaECal",
                         "TauTrack.deta",
                         "TauTrack.pt",
                         "TauTrack.jetpt", 
                         #"TauTrack.d0TJVA",
                         #"TauTrack.d0SigTJVA",
                         #"TauTrack.z0sinthetaTJVA",
                         #"TauTrack.z0sinthetaSigTJVA", 
                        ],
            "NeutralPFO": ["NeutralPFO.dphiECal",
                           "NeutralPFO.dphi",
                           "NeutralPFO.detaECal",
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
                           # "NeutralPFO.NHitsInEM1",  #
                           "NeutralPFO.ptSubRatio",  #
                           "NeutralPFO.energyfrac_EM2",  #
                           #"NeutralPFO.energy_EM1",
                           #"NeutralPFO.energy_EM2",
                           "NeutralPFO.EM1CoreFrac",
                           "NeutralPFO.firstEtaWRTClusterPosition_EM1",
                           #"NeutralPFO.firstEtaWRTClusterPosition_EM2",
                           "NeutralPFO.secondEtaWRTClusterPosition_EM1",
                           "NeutralPFO.secondEtaWRTClusterPosition_EM2",
                           ],
            "ShotPFO": ["ShotPFO.dphiECal",
                        "ShotPFO.dphi",
                        "ShotPFO.detaECal",
                        "ShotPFO.deta",
                        "ShotPFO.pt",
                        "ShotPFO.jetpt", ],
            "ConvTrack": ["ConvTrack.dphiECal",
                          "ConvTrack.dphi",
                          "ConvTrack.detaECal",
                          "ConvTrack.deta",
                          "ConvTrack.pt",
                          "ConvTrack.jetpt", 
                          #"ConvTrack.d0TJVA",
                          #"ConvTrack.d0SigTJVA",
                          #"ConvTrack.z0sinthetaTJVA",
                          #"ConvTrack.z0sinthetaSigTJVA", 
                         ],
            "Label": ["TauJets.truthDecayMode"],
        }

        self._dataset.sel_vars = ["TauJets.truthDecayMode", "TauJets.mcEventNumber"]

        # don't modify this
        self._dataset.dtype = {
            "TauTrack": np.float32,
            "NeutralPFO": np.float32,
            "ShotPFO": np.float32,
            "ConvTrack": np.float32,
            "Label": np.long,
        }

        # number of objects per tau
        self._dataset.n_steps = {
            "TauTrack": 3,
            "NeutralPFO": 8,
            "ShotPFO": 6,
            "ConvTrack": 4,
            "Label": None,
        }

        self._dataset.log_vars = {
            "TauTrack.pt": 1e-3,
            "TauTrack.jetpt": 1e-3,
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
        
        self._dataset.atan_vars = {
            "NeutralPFO.ptSubRatio", 
#             "NeutralPFO.firstEtaWRTClusterPosition_EM1",
#             "NeutralPFO.firstEtaWRTClusterPosition_EM2",
#             "TauTrack.d0TJVA",
#             "TauTrack.d0SigTJVA",
#             "TauTrack.z0sinthetaTJVA",
#             "TauTrack.z0sinthetaSigTJVA", 
#             "ConvTrack.d0TJVA",
#             "ConvTrack.d0SigTJVA",
#             "ConvTrack.z0sinthetaTJVA",
#             "ConvTrack.z0sinthetaSigTJVA", 
        }

        self._dataset.branch = ["TauTrack", "NeutralPFO", "ShotPFO", "ConvTrack", "Label"]

        self._dataset.shape = dict()
        for name in self._dataset.branch:
            if name == "Label":
                self._dataset.shape[name] = ()  # HC ! Be careful !
            else:
                self._dataset.shape[name] = (self._dataset.n_steps[name], len(self._dataset.data[name]))
