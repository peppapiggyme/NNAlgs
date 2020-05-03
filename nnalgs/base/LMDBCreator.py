import os
import pathlib
from abc import abstractmethod

import lmdb
import uproot

from nnalgs.utils.Common import write_json, read_json
from nnalgs.utils.Logger import get_logger


class BaseLMDBCreator(object):
    """
    Base class for all LMDB database creators
    """

    def __init__(self, obj_name, paths, tree_name, lmdb_dir, json_dir, mode):
        """
        :param obj_name: name of the creator
        :param paths: path to the ROOT file
        :param tree_name: name of the ROOT tree
        :param lmdb_dir: path to the LMDB database
        :param json_dir: path to the JSON file
        :param mode: dataset mode, 'Train' or 'Validation'
        """
        self.paths = paths
        self.obj_name = obj_name
        self.tree_name = tree_name
        self.lmdb_dir = lmdb_dir
        self.json_dir = json_dir
        self.mode = mode

        # hash table cache
        self._cache = {}

        # super counter
        self._counter = 0

        # uproot
        self._root_file = None
        self._root_tree = None

        # pre-processing
        self._variables = []  # names, scales, offsets, ... -> part of variables.json
        self._preproc = {}  # names: size, mean, std -> preproc.json

        # a specific logger for the class
        self._logger = get_logger(self.obj_name, 'INFO')

        # lmdb.Environment
        self._env = self._create_env()

    # Pre-processing
    @abstractmethod
    def _save_to_json(self):
        """
        This will create the json file that contains scales and offsets
        Later the file can be used by lwtnn
        :return:
        """
        raise NotImplementedError

    def _execute_preproc(self):
        """
        The pre-processing method runs before the creation
        :return:
        """
        for path in self.paths:
            self._loop_preproc_path(path)

        self._logger.info(f"\n - Pre-processing finito!\n")

    def _loop_preproc_path(self, path):
        """
        One iteration over the multiple root files
        :param path:
        :return:
        """
        chunks = self._get_chunks_from_root(path)
        for chunk in chunks:
            self._loop_preproc_chunk(chunk)

    @abstractmethod
    def _loop_preproc_chunk(self, chunk):
        """
        One iteration over the chunks in ROOT file
        :param chunk:
        :return:
        """
        raise NotImplementedError

    # LMDB creation
    def _create_env(self):
        try:
            return lmdb.open(self.lmdb_dir, map_size=1024 ** 4, create=True)
        except Exception as e:
            self._logger.error(f"Failed to create LMDB environment: {e!r}")
            return None

    def _get_chunks_from_root(self, path):
        try:
            self._root_file = uproot.open(path)
            self._root_tree = self._root_file[self.tree_name]
            chunks = list(self._root_tree.clusters())
            self._logger.info(f"Processing: {path}")
            self._logger.info(
                f" - NEntries = NChunks x Length: "
                f"{self._root_tree.numentries} = {len(chunks)} x {chunks[0][1] - chunks[0][0]}"
            )
            return chunks
        except Exception as e:
            self._logger.error(f"Failed to get chunks from ROOT file: {path}")
            self._logger.error(f"Error: {e!r}")

    def _write_cache(self):
        """
        lmdb cache writing
        :return:
        """
        with self._env.begin(write=True) as txn:
            for k, v in self._cache.items():
                txn.put(k.encode(), v)

        self._cache = {}

    def _execute_create(self):
        """
        The method that creates the database
        :return:
        """
        for path in self.paths:
            self._loop_create_path(path)

        self._logger.info(f"\n - Processed {self._counter} samples in total!\n")

    def _loop_create_path(self, path):
        """
        One iteration over the multiple root files
        :param path:
        :return:
        """
        chunks = self._get_chunks_from_root(path)
        for chunk in chunks:
            self._loop_create_chunk(chunk)

    @abstractmethod
    def _loop_create_chunk(self, chunk):
        """
        One iteration over the chunks in ROOT file
        :param chunk:
        :return:
        """
        raise NotImplementedError

    # class tools
    @abstractmethod
    def _get_removed_indices(self, df):
        """
        Apply selection ^o^
        :param df: pandas.DataFrame
        :return: list of indices to be removed
        """
        raise NotImplementedError

    @abstractmethod
    def _preprocessing_decaymode(self, ft, arr):
        """
        Pre-process array at creation time
        :param ft: feature name
        :param arr: numpy.ndarray
        :return: pre-processed numpy.ndarray
        """
        raise NotImplementedError

    # json
    def _write_json(self, final_dict):
        pathlib.Path(f"{self.json_dir}").mkdir(parents=True, exist_ok=True)
        variables = os.path.join(f"{self.json_dir}", "variables.json")
        preproc = os.path.join(f"{self.json_dir}", "preproc.json")
        write_json(final_dict, variables)
        write_json(self._preproc, preproc)
        self._logger.info(f"Saved to {variables} (for lwtnn) and {preproc} (for reading in testing).")

    def _read_json(self):
        preproc = os.path.join(f"{self.json_dir}", "preproc.json")
        return read_json(preproc)

    # interface
    def execute(self):
        if self.mode == "Train":
            self._logger.info(r">>> Pre-processing ...")
            self._execute_preproc()
            self._logger.info(f">>> Saving JSON file to {self.json_dir}...")
            self._save_to_json()
        self._logger.info(f">>> Creating LMDB database in {self.lmdb_dir} ...")
        self._execute_create()

    # extent closing
    def close(self):
        self._counter = 0
        self._cache.clear()
        self._env.close()
        self._root_file = None
        self._root_tree = None
        self._variables.clear()
        self._preproc.clear()
