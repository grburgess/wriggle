import collections
import os
from pathlib import Path

import h5py
import numpy as np
import yaml
from threeML import OGIPLike, silence_warnings

silence_warnings()


def sanitize_filename(filename, abspath=False):

    sanitized = os.path.expandvars(os.path.expanduser(filename))

    if abspath:

        return os.path.abspath(sanitized)

    else:

        return sanitized


class GRBDatum(object):
    def __init__(
        self,
        name,
        observation,
        background,
        background_error,
        response,
        mc_energies,
        ebounds,
        exposure,
        mask,
        significance,
    ):

        self._n_chans = len(observation)

        assert self._n_chans == len(background)
        assert self._n_chans == response.shape[0]
        assert self._n_chans == len(background_error)
        assert self._n_chans == len(mask)

        assert sum(mask) > 0

        assert exposure > 0

        self._name = name
        self._observation = np.array(observation).astype(int)
        self._background = np.array(background)
        self._background_error = background_error
        self._response = response

        self._response = response
        self._ebounds = ebounds
        self._mc_energies = mc_energies

        self._exposure = exposure
        self._mask = mask
        self._significance = significance

        if self._name.startswith('n'):

            self._det_type: int = 1

        elif self._name.startswith('b'):

            self._det_type = 2

        else:

            raise RuntimeError()

        
    @property
    def name(self):
        return self._name

    @property
    def det_type(self) -> int:
        return self._det_type

    
    @property
    def response(self):
        return self._response

    @property
    def response_transpose(self):
        return self._response.T

    @property
    def observation(self):
        return self._observation

    @property
    def background(self):
        return self._background

    @property
    def background_error(self):
        return self._background_error

    @property
    def mask(self):
        return self._mask

    @property
    def mask_stan(self):
        return np.where(self._mask)[0] + 1

    @property
    def n_channels_used(self):
        return sum(self._mask)  # CHECK THIS MOTHER FUCKER

    @property
    def idx_background_zero(self):
        return np.where(self._background[self._mask] == 0)[0]

    @property
    def idx_background_nonzero(self):
        return np.where(self._background[self._mask] > 0)[0]

    @property
    def n_bkg_zero(self):
        return sum(self._background[self._mask] == 0)

    @property
    def n_bkg_nonzero(self):
        return sum(self._background[self._mask] > 0)

    @property
    def n_chans(self):
        return self._n_chans

    @property
    def n_echans(self):
        return self._response.shape[1]

    @property
    def exposure(self):
        return self._exposure

    @property
    def significance(self):
        return self._significance

    @property
    def ebounds(self):
        return self._mc_energies

    @property
    def ebounds_lo(self):
        return self._mc_energies[:-1]

    @property
    def ebounds_hi(self):
        return self._mc_energies[1:]

    @property
    def cbounds(self):
        return self._ebounds

    @property
    def cbounds_lo(self):
        return self._ebounds[:-1]

    @property
    def cbounds_hi(self):
        return self._ebounds[1:]

    @classmethod
    def from_ogip(cls, name, obs_file, bkg_file, rsp, selection, spectrum_number=1):
        """
        Create the base data from FITS files.

        :param cls:
        :param name:
        :param obs_file:
        :param bkg_file:
        :param rsp:
        :param selection:
        :param spectrum_number:
        :returns:
        :rtype:

        """

        # use 3ml to read the
        # FITS files
        ogip = OGIPLike(
            name,
            observation=obs_file,
            background=bkg_file,
            response=rsp,
            spectrum_number=spectrum_number,
            verbose=False,
        )

        # set the mask

        ogip.set_active_measurements(*selection)

        return cls(
            name,
            ogip.observed_counts,
            ogip.background_counts,
            ogip.background_count_errors,
            ogip.response.matrix,
            ogip.response.monte_carlo_energies,
            ogip.response.ebounds,
            ogip.exposure,
            ogip.mask,
            ogip.significance,
        )

    def to_hdf5_file_or_group(self, name):
        """
        write the data to and HDF5 file or group

        :param name:
        :returns:
        :rtype:

        """

        if isinstance(name, h5py.File) or isinstance(name, h5py.Group):

            is_file = False
            f = name

        else:

            if_file = True
            f = h5py.File(name, "w")

        f.attrs["name"] = self._name
        f.attrs["exposure"] = self._exposure
        f.attrs["significance"] = self._significance

        f.create_dataset(
            "observation", data=self._observation, compression="lzf", shuffle=True
        )
        f.create_dataset(
            "background", data=self._background, compression="lzf", shuffle=True
        )
        f.create_dataset(
            "background_error",
            data=self._background_error,
            compression="lzf",
            shuffle=True,
        )
        f.create_dataset(
            "response", data=self._response, compression="lzf", shuffle=True
        )
        f.create_dataset(
            "mc_energies", data=self._mc_energies, compression="lzf", shuffle=True
        )
        f.create_dataset("ebounds", data=self._ebounds, compression="lzf", shuffle=True)
        f.create_dataset("mask", data=self._mask, compression="lzf", shuffle=True)

        if is_file:

            f.close()

    @classmethod
    def from_hdf5_file_or_group(cls, name):
        """
        read in the data from an HDF5 file

        :param cls:
        :param name:
        :returns:
        :rtype:

        """

        # check if we are from a bigger file or opening one
        if isinstance(name, h5py.File) or isinstance(name, h5py.Group):

            # keep track if we will need to close
            is_file = False
            f = name

        else:

            if_file = True
            f = h5py.File(name, "r")

        # extract all the shit

        name = f.attrs["name"]
        exposure = f.attrs["exposure"]
        significance = f.attrs["significance"]

        observation = f["observation"][()]
        background = f["background"][()]
        background_error = f["background_error"][()]
        response = f["response"][()]
        mc_energies = f["mc_energies"][()]
        ebounds = f["ebounds"][()]
        mask = f["mask"][()]

        if is_file:
            f.close()

        return cls(
            name,
            observation,
            background,
            background_error,
            response,
            mc_energies,
            ebounds,
            exposure,
            mask,
            significance,
        )


class GRBInterval(object):
    def __init__(self, grb_name: str, *data):
        """
        A time interval consisting of all the detectors

        :param name:
        :param observations:
        :param backgrounds:
        :param responses:
        :returns:
        :rtype:

        """

        self._data = collections.OrderedDict()
        self._n_dets: int = len(data)

        n_chans = []
        n_echans = []

        for datum in data:

            assert isinstance(datum, GRBDatum)

            self._data[datum.name] = datum
            n_echans.append(datum.n_echans)
            n_chans.append(datum.n_chans)

        self._name: str = grb_name

        self._n_chans: int = max(n_chans)
        self._n_echans: int = max(n_echans)

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self):

        return self._data

    @property
    def n_detectors(self) -> int:

        return self._n_dets

    @property
    def n_echans(self) -> int:
        return self._n_echans

    @property
    def n_chans(self) -> int:
        return self._n_chans

    @classmethod
    def from_yaml(cls, file_name):
        """

        Construct from a yaml file that specifies
        where the PHA/FITS files are. This is to
        initially build the data sets and then dump
        them to HDF5. The yaml file should look like

        grb_name: grb1
        dir: ~/home/location
        spectrum_number: 1   
        
         detectors:
            n1:
               - 10-900
            b0:
               - 250-30000
        

        :param cls:
        :param file_name:
        :returns:
        :rtype:

        """

        # open the file

        with open(file_name, "r") as f:

            # read the shit in from the yaml file

            d = yaml.load(f, Loader=yaml.SafeLoader)

            # call the dict construction method

        return cls.from_dict(d)


    
    @classmethod
    def from_dict(cls, d):
        """
        create from a dictionary that is initially
        from the yaml file and will trigger the reading
        of the OGIP files

        :param cls:
        :param d:
        :param spectrum_number:
        :returns:
        :rtype:

        """

        dets = d["detectors"]

        grb_name = d["grb_name"]

        spectrum_number = int(d["spectrum_number"])
        
        directory = Path(sanitize_filename(d["dir"]))

        data = []

        # we always want the sorted
        sorted_dets = list(dets.keys())
        sorted_dets.sort()

        # this depends on the file
        # structure. This is not great
        # but not bad

        for det in sorted_dets:

            # match pha

         
            obs_file = [x for x in directory.glob(f"*{det}*.pha") if "bak" not in str(x)][0]
            
            
            # match bak
            bak_file = [x for x in directory.glob(f"*{det}*bak.pha")][0]

            # match rsp
            rsp_file = [x for x in directory.glob(f"*{det}*.rsp")][0]

            datum = GRBDatum.from_ogip(
                det, obs_file, bak_file, rsp_file, dets[det], spectrum_number
            )

            data.append(datum)
        return cls(grb_name, *data)

    def to_hdf5_file_or_group(self, name):
        """
        write the interval to HDF5 which triggers the recursive
        writers

        :param name:
        :returns:
        :rtype:

        """

        if isinstance(name, h5py.File) or isinstance(name, h5py.Group):

            is_file = False
            f = name

        else:

            if_file = True
            f = h5py.File(name, "w")

        for k, v in self._data.items():

            grp = f.create_group(k)
            grp.attrs["grb_name"] = self._name

            v.to_hdf5_file_or_group(name=grp)

        if is_file:

            f.close()

    @classmethod
    def from_hdf5_file_or_group(cls, name):
        """FIXME! briefly describe function

        :param cls:
        :param name:
        :returns:
        :rtype:

        """

        if isinstance(name, h5py.File) or isinstance(name, h5py.Group):

            is_file = False
            f = name

        else:

            if_file = True
            f = h5py.File(name, "r")

        data = []

        grb_name = f.attrs["grb_name"]

        for det in f.keys():

            datum = GRBDatum.from_hdf5_file_or_group(f[det])

            data.append(datum)

        if is_file:

            f.close()

        return cls(grb_name, *data)

    def to_stan_dict(self, k: int =25):

        stan_data = collections.OrderedDict()


        stan_data["N_echan"] = self._n_echans
        stan_data["N_chan"] = self._n_chans

        observed_counts = np.zeros(
            (self._n_dets, self._n_chans)
        )

        background_counts = np.zeros(
            ( self._n_dets, self._n_chans)
        )

        background_errors = np.zeros(
            (self._n_dets, self._n_chans)
        )

        idx_background_zero = np.zeros(
            (self._n_dets, self._n_chans)
        )

        idx_background_nonzero = np.zeros(
            (self._n_dets, self._n_chans)
        )
        
        n_bkg_zero = np.zeros(self._n_dets)

        n_bkg_nonzero = np.zeros(self._n_dets)

        responses = np.zeros((self._n_dets,
                              self._n_chans,
                              self._n_echans)
        )

        exposures = np.zeros(self._n_dets)

        n_echan = np.zeros(self._n_dets)

        n_chan = np.zeros(self._n_dets)

        masks = np.zeros((self._n_dets, self._n_chans))

        n_channels_used = np.zeros(self._n_dets)

        ebounds_lo = np.zeros((self._n_dets, self._n_echans))

        ebounds_hi = np.zeros((self._n_dets, self._n_echans))

        cbounds_lo = np.zeros((self._n_dets, self._n_chans))

        cbounds_hi = np.zeros((self._n_dets, self._n_chans))

        total_number_of_channels_used = 0

        det_type = np.zeros(self._n_dets, dtype=int)
        
        for j, (det, datum) in enumerate(self._data.items()):

            observed_counts[j, : datum.n_chans] = datum.observation
            background_counts[j, : datum.n_chans] = datum.background

            idx_background_zero[j, : datum.n_bkg_zero] = (
                datum.idx_background_zero + 1
            )
            idx_background_nonzero[j, : datum.n_bkg_nonzero] = (
                datum.idx_background_nonzero + 1
            )

            n_bkg_zero[j] = datum.n_bkg_zero
            n_bkg_nonzero[j] = datum.n_bkg_nonzero

            background_errors[j, : datum.n_chans] = datum.background_error


            det_type[j] = datum.det_type
            
            
            # correct format for STAN
            
            responses[j, :datum.n_chans, :datum.n_echans] = datum.response

            this_mask = datum.mask_stan

            masks[j, : len(this_mask)] = this_mask

            # this could be a bug!
            n_channels_used[j] = datum.n_channels_used

            n_chan[j] = datum.n_chans
            n_echan[j] = datum.n_echans

            ebounds_lo[j, : datum.n_echans] = datum.ebounds_lo
            ebounds_hi[j, : datum.n_echans] = datum.ebounds_hi

            cbounds_lo[j, : datum.n_chans] = datum.cbounds_lo
            cbounds_hi[j, : datum.n_chans] = datum.cbounds_hi

            exposures[j] = datum.exposure

            total_number_of_channels_used += datum.n_channels_used


        stan_data["N_dets"] = self._n_dets

        stan_data["det_type"] = det_type.astype(int)

        stan_data["observed_counts"] = observed_counts
        stan_data["background_counts"] = background_counts
        stan_data["background_errors"] = background_errors
        stan_data["idx_background_nonzero"] = idx_background_nonzero.astype(int)
        stan_data["idx_background_zero"] = idx_background_zero.astype(int)
        stan_data["N_bkg_zero"] = n_bkg_zero.astype(int)
        stan_data["N_bkg_nonzero"] = n_bkg_nonzero.astype(int)
        stan_data["N_chan"] = np.max(n_chan.astype(int))
        stan_data["N_echan"] =np.max( n_echan.astype(int))

        stan_data["ebounds_lo"] = ebounds_lo
        stan_data["ebounds_hi"] = ebounds_hi
        stan_data["cbounds_lo"] = cbounds_lo
        stan_data["cbounds_hi"] = cbounds_hi
        stan_data["exposure"] = exposures
        stan_data["response"] = responses
        stan_data["mask"] = masks.astype(int)
        stan_data["N_channels_used"] = n_channels_used.astype(int)
        stan_data["max_range"] = np.max(ebounds_hi) - np.min(ebounds_lo)

        stan_data["k"] = k
        
        return stan_data

    

