from abc import ABCMeta

import awkward
import numpy as np

from nnalgs.base.LMDBCreator import BaseLMDBCreator
from nnalgs.utils.Math import combine_mean_std


class DecayModeLMDBCreator(BaseLMDBCreator, metaclass=ABCMeta):
    """
    Class for decay mode classification LMDB database creators
    """

    def __init__(self, sel_vars, data, n_steps, dtype, log_vars, atan_vars, **kwargs):
        """
        :param sel_vars: a list of the variables for selection (e.g. TauJet variables)
        :param data: a dict of all the inputs and outputs variables for each input branch
        :param n_steps: a dict of the maximum time steps to be stored for each input branch
        :param dtype: a dict of NumPy dtype to be stored for each input branch
        :param trans: boolean, do pre-processing or not
        :param kwargs: common arguments for base class
        """
        self.sel_vars = sel_vars
        self.data = data
        self.n_steps = n_steps
        self.dtype = dtype
        self.log_vars = log_vars
        self.atan_vars = atan_vars

        super().__init__(**kwargs)

    def _loop_create_chunk(self, chunk):
        entry_start, entry_stop = chunk

        # do selection (train, test split based on mc event number, 2020/03/24 -> full set training)
        self._logger.debug(f"   - Chunk here is: {entry_start} ~ {entry_stop}")
        df_sel = self._root_tree.pandas.df(self.sel_vars, entrystart=entry_start, entrystop=entry_stop)

        # 2020/03/24 -> nothing
        removed_indices = self._get_removed_indices(df=df_sel)
        counter_here = self._counter

        # loop over each branch: Label, PFOs, Track, ...
        for name, vars in self.data.items():
            self._logger.debug(f"     - Branch here is: {name}")

            # Overall counter! Till the end!
            self._counter = counter_here
            df = self._root_tree.pandas.df(vars, entrystart=entry_start, entrystop=entry_stop, flatten=False)

            # apply selection based on indices
            df.drop(removed_indices, inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Label shape is easy to deal with
            if name == "Label":
                assert len(vars) == 1
                arr = np.asarray(df, dtype=self.dtype[name])
                arr = arr.reshape(len(arr))

            # Some variables added for testing purpose
            elif name == "Inspector":
                arr = np.zeros((len(df), len(self.data[name])), dtype=self.dtype[name])
                for i, var in enumerate(self.data[name]):
                    arr[:, i] = np.asarray(df[var], dtype=self.dtype[name])

            # In this case, all others are 3-D sequences
            else:
                # get array
                longest = max([len(a) for a in df[vars[0]]])  # take the first variable as reference
                list_here = list()
                for rec in df.to_records():
                    # index = rec[0]
                    list_vars = [rec[i + 1] for i in range(len(vars))]
                    vars_data = np.asarray(list_vars, dtype=self.dtype[name]).transpose()
                    len_seq = vars_data.shape[0]
                    # zero padding of sequence (to save everything in one NumPy array)
                    vars_data = np.pad(vars_data, ((0, longest - len_seq), (0, 0)), 'constant')
                    list_here.append(vars_data)
                arr = np.asarray(list_here, dtype=self.dtype[name])

                for index, var in enumerate(vars):
                    arr[:, :, index] = self._preprocessing_decaymode(var, arr[:, :, index])

                # NOTE: should we save as much as possible and then having room to tune the seq max_len?
                if self.n_steps[name] < longest:
                    arr = arr[:, :self.n_steps[name], :]
                if self.n_steps[name] > longest:
                    self._logger.warning(
                        f"required number of objects {self.n_steps[name]} is more than the actual "
                        f"maximum number of objects {longest}, will NOT fill with zeros!!!"
                    )
                    # if still want this feature, uncomment the line below
                    arr = np.pad(arr, ((0, 0), (0, 0), (0, self.n_steps[name] - longest)))

            # do some testing here if you want
            self._logger.debug(arr[:3])
            self._logger.debug(arr.shape)

            # write the cache into database
            # like this {"NeutralPFO-019961013": array([1,2,3, ..., batch_size])}
            for i in range(arr.shape[0]):
                if name != "Label" and arr[i].shape != (self.n_steps[name], len(vars)):
                    self._logger.warning("{}-{:09d} has unexpected shape {}! This might cause crash at training time!".format(name, self._counter, arr[i].shape))
                key = "{}-{:09d}".format(name, self._counter)
                self._cache[key] = arr[i].copy(order='C')  # C
                if i % 1000 == 0:
                    self._write_cache()
                    self._logger.debug(f"       - Cache --> counter={self._counter}, i={i} is written")
                self._counter += 1

    def _loop_preproc_chunk(self, chunk):
        """
        The pre-processing strategy is evaluated on full samples.
        No selection is required (yet not implemented)
        """
        entry_start, entry_stop = chunk

        for name, fts in self.data.items():

            # Skip Label ...
            if name == "Label":
                continue

            # loop over fts -> features of each branch
            for ft in fts:

                # in this case, i.e. multi-obj in a tau, uproot will return a JaggedArray via awkward package
                array: awkward.JaggedArray = self._root_tree.array(ft, entrystart=entry_start, entrystop=entry_stop)

                # now point array to the flatten NumPy array
                array: np.ndarray = array.flatten()

                # check log and ATan transformation
                ft, array = self._atan_tans(ft, array)

                # Maybe don't need to use the same dtype (-_-)
                array.flatten().dtype = self.dtype[name]  # <- np.float32 ...

                # approximation of mean and std, for the scale and offset in pre-processing
                size_here, mean_here, std_here = entry_stop - entry_start, array.mean(), array.std()

                # If it is not the first chunk
                if ft in self._preproc and set(self._preproc[ft].keys()) == {"size", "mean", "std"}:
                    l_n = [self._preproc[ft]["size"], size_here]
                    l_mean = [self._preproc[ft]["mean"], mean_here]
                    l_std = [self._preproc[ft]["std"], std_here]
                    # a useful function which can combine a list of samples
                    size_here, mean_here, std_here = combine_mean_std(l_n, l_mean, l_std)
                self._preproc[ft] = dict(size=size_here, mean=mean_here, std=std_here)

        self._logger.debug(self._preproc, "\n")

    def _save_to_json(self):
        for name, fts in self.data.items():
            if name == "Label":
                continue
            d_fts = {'name': name, 'variables': []}
            for ft, stat in self._preproc.items():
                if name in ft:

                    # prior knowledge -> self._preproc is consistent with self._variables
                    if ft.endswith(".phi"):
                        self._preproc[ft]["mean"], self._preproc[ft]["std"] = 0.0, 0.5 * 3.1415926
                    elif ft.endswith(".eta"):
                        self._preproc[ft]["mean"], self._preproc[ft]["std"] = 0.0, 0.5 * 2.5000000
                    elif ft.endswith(".dphi") or ft.endswith(".deta"):
                        self._preproc[ft]["mean"] = 0.0
                    elif ft.endswith(".dphiECal") or ft.endswith(".detaECal"):
                        self._preproc[ft]["mean"] = 0.0
                    # no prior knowledge
                    else:
                        pass

                    d_fts['variables'].append(
                        dict(name=ft.split('.')[-1], offset=float(-1 * stat["mean"]), scale=float(0.5 / stat["std"])))

            self._variables.append(d_fts)

        outputs = [
            {
                'name': 'classes',
                'labels': ['c_1p0n', 'c_1p1n', 'c_1pXn', 'c_3p0n', 'c_3pXn']
            }
        ]
        final_dict = {
            'input_sequences': self._variables,
            'inputs': [],
            'outputs': outputs
        }

        # Saving
        self._write_json(final_dict)

    def _get_removed_indices(self, df):
        """
        The performance is not as bad as i looks like...
        :param df: pandas.DataFrame
        :return: list of indices
        For cross validation, one can change the remainer = 0, 1, 2, 3, 5 (5-fold)
        """
        if self.mode == 'Train':
            removed_indices = df[(df["TauJets.truthDecayMode"] > 4) | (df["TauJets.mcEventNumber"] % 5 == 0)].index  # 4/5 of all
        elif self.mode == 'Validation':
            removed_indices = df[(df["TauJets.truthDecayMode"] > 4) | (df["TauJets.mcEventNumber"] % 5 != 0)].index  # 1/5 of all
        elif self.mode == 'Test':
            removed_indices = df[(df["TauJets.truthDecayMode"] > 4) | (df["TauJets.mcEventNumber"] % 5 != 0)].index  # not used
        else:
            raise ValueError(f"Cannot set type as {self.mode}")

        return removed_indices

    def _preprocessing_decaymode(self, ft, arr):
        """
        Pre-process array at creation time
        :param ft: feature name
        :param arr: numpy.ndarray
        :return: pre-processed numpy.ndarray
        """
        preproc = self._read_json()

        # NOTE: arrays here are padded with zeros (!)
        masked = np.ma.masked_equal(arr, 0)

        # Naming is consistent with those in self._preproc
        ft, masked = self._atan_tans(ft, masked)

        offset, scale = preproc[ft]["mean"], 0.5 / preproc[ft]["std"]
        masked = np.multiply(np.subtract(masked, offset), scale)

        return masked.filled(0)

    def _atan_tans(self, ft, arr):
        """check log and atan transformation"""

        if ft in self.log_vars.keys():
            arr = np.log10(np.maximum(arr, self.log_vars[ft]))
            ft = ''.join((ft, '_log'))
        elif ft in self.atan_vars:
            arr = np.arctan(arr)
            ft = ''.join((ft, '_atan'))

        return ft, arr
