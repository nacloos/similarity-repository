import os
import pickle

import numpy as np
import xarray as xr
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_project_api import EcephysProjectWarehouseApi
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine

from mouse_vision.core.utils import open_dataset
from mouse_vision.core.default_dirs import \
    NEURAL_RESULTS_DIR, \
    MANIFEST_PATH, \
    NEUROPIX_DATA_PATH, \
    NEUROPIX_DATA_PATH_WITH_RELS
from mouse_vision.core.constants import \
    TIME_RESOLUTION, \
    TIME_BINS, \
    VISUAL_AREAS, \
    NUM_NS_STIM, \
    NUM_NS_STIM_PLUS_GREY, \
    NUM_TRIALS, \
    NUM_TIMEPOINTS

class NeuropixelsDataProcessor():
    def __init__(self, manifest_path):
        self.cache = self._get_ecephys_project_cache(manifest_path)
        self.bo_sessions = self._get_brain_observatory_type_sessions()

    def _get_ecephys_project_cache(self, manifest_path):
        """
        Gets the electrophysiology project cache for downloading data.

        Inputs:
            manifest_path : (string) path to manifest.json

        Outputs:
            cache         : (EcephysProjectCache) cache that has information about
                            the experiment sessions.
        """
        cache = EcephysProjectCache(
            manifest=manifest_path,
            fetch_api=EcephysProjectWarehouseApi(
                RmaEngine(
                    scheme="http",
                    host="api.brain-map.org",
                    timeout=50*60  # Set download timeout to 50 minutes
                )
            )
        )

        return cache

    def _get_brain_observatory_type_sessions(self):
        """
        Given an EcephysProjectCache, return the pandas data frame that has information
        about each session in the data.

        Inputs:
            cache : (EcephysProjectCache) cache obtained from the manifest.json file

        Outputs:
            brain_observatory_type_sessions : (pandas.DataFrame) data frame containing information
                                              about each session in the cache belonging to brain
                                              observatory sessions.
        """
        assert isinstance(self.cache, EcephysProjectCache)
        assert "brain_observatory_1.1" in self.cache.get_all_session_types()

        sessions = self.cache.get_session_table()
        brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]

        return brain_observatory_type_sessions

    def get_frame_presentation_ids(self, session, frame_id, stim_type):
        """
        Get stimulus presentation ID data frame for this session and then get presentation
        IDs associated with a particular condition id.

        Inputs:
            session   : (EcephysSession) data for a particular session
            frame_id  : (int) frame ID of the stimulus
            stim_type : (string) desired stimulus type

        Outputs:
            stim_pres_ids : (numpy.ndarray) numpy array of presentation IDs associated with
                            the frame_id
        """
        assert isinstance(session, EcephysSession)

        # Get stimulus presentation ID data frame for this session
        stim_pres_df = session.get_stimulus_table(stim_type)

        # Extract presentation IDs associated with frame_id
        stim_pres_ids = stim_pres_df[stim_pres_df["frame"] == frame_id]
        stim_pres_ids = stim_pres_ids.index.values

        return stim_pres_ids

    def get_spike_counts_for_condition(self, session, cond_pres_ids, units):
        """
        Acquires an xarray containing spike counts for the given condition ID across all
        presentations and desired units.

        Inputs:
            session       : (EcephysSession)
            cond_pres_ids : (numpy.ndarray) list of presentation IDs associated with the frame_id
            units         : (pandas.DataFrame) data frame with information about each unit such as
                            visual area and unit id.

        Outputs:
            spike_counts  : (xarray.DataArray) this data array is 3-dimensional. The dimension names
                            are: ('stimulus_presentation_id', 'time_relative_to_stimulus_onset', 'unit_id')
                            It is of dimensions (num_trials, 25, num_units).
        """
        assert isinstance(session, EcephysSession)
        assert isinstance(units, pd.DataFrame)

        spike_counts = session.presentationwise_spike_counts(
            stimulus_presentation_ids=cond_pres_ids,
            bin_edges=TIME_BINS,
            unit_ids=units.index.values
        )

        return spike_counts

    def add_metadata_to_spike_count_array(self, session_id, spike_counts, frame_id, current_specimen_id, units):
        """
        This function adds additional metadata to the units dimension since we want to know
        which specimen and visual area the unit is from.

        Inputs:
            session_id          : (numpy.int64) the ID of the session
            spike_counts        : (xarray.DataArray) contains the data obtained from the
                                  Allen SDK API. Dimensions are:
                                  ('stimulus_presentation_id', 'time_relative_to_stimulus_onset', 'unit_id')
            frame_id            : (numpy.int64) the ID of the stimulus (between -1 and 117 inclusive).
                                  Presumably, frame id of -1 is the blank stimulus.
            current_specimen_id : (numpy.int64) the ID of the current specimen
            units               : (pandas.DataFrame) data frame containing information about
                                  the desired units (e.g. visual area)

        Outputs:
            frs : (xarray.DataArray) new data structure containing additional metadata and
                  spike counts converted to firing rates.
        """
        assert isinstance(spike_counts, xr.DataArray)
        assert isinstance(frame_id, np.int64)
        assert isinstance(current_specimen_id, np.int64)
        assert len(spike_counts.dims) == 3

        stim_presentation_id = spike_counts.stimulus_presentation_id.values
        if int(stim_presentation_id.size) < NUM_TRIALS:
            print(f"[WARNING] Condition {frame_id} for specimen {current_specimen_id}"
                   f" in session {session_id} has less than {NUM_TRIALS} trials ({stim_presentation_id.size} trials).")
            # If less than 50 trials, append nans to presentation ids and also add
            # NaNs to the spike count data array.
            n_missing = NUM_TRIALS - int(stim_presentation_id.size)
            nan_array = np.ones(n_missing) * np.NaN
            stim_presentation_id = np.append(stim_presentation_id, nan_array)

            # Create NaN matrix for remaining trials
            n_time = spike_counts.values.shape[1]
            n_units = spike_counts.values.shape[2]
            nan_spike_counts = np.ones((n_missing, n_time, n_units)) * np.NaN

            # Append to original spike counts matrix
            spike_counts_with_nans = np.append(spike_counts.values, nan_spike_counts, axis=0)
            expand_spike_counts = np.expand_dims(spike_counts_with_nans, axis=0)

        elif int(stim_presentation_id.size) == NUM_TRIALS:
            # Increase number of dimensions (axis 0 dimension will be the condition ID dimension)
            # Final spike counts array shape: 1 x trial x time x units
            expand_spike_counts = np.expand_dims(spike_counts.values, axis=0)
        else:
            raise ValueError(f"[ERROR] Condition {frame_id} for specimen {current_specimen_id} in "
                        f"session {session_id} has more than {NUM_TRIALS} trials ({stim_presentation_id.size} trials).")

        assert expand_spike_counts.shape[1] == NUM_TRIALS # 50 trials
        time = spike_counts.time_relative_to_stimulus_onset.values
        current_specimen_id = np.repeat(current_specimen_id, spike_counts.unit_id.values.size)
        unit_ids = spike_counts.unit_id.values
        assert np.array_equal(unit_ids, units.index.values)

        # Here we are adding additional metadata to each unit, which includes unit_id,
        # specimen_id and visual_area since we may want to slice along these dimensions
        # when doing neural fitting. We use pandas.MultiIndex here.
        unit_id_multi_index = pd.MultiIndex.from_arrays(
            [
                unit_ids,
                current_specimen_id,
                units["ecephys_structure_acronym"].values,
                units["image_selectivity_ns"].values,
                units["run_pval_ns"].values
            ],
            names=("unit_id", "specimen_id", "visual_area", "image_selectivity_ns", "run_pval_ns")
        )

        # Now construct a new xarray.DataArray with additional data for each unit
        spike_counts = xr.DataArray(
            expand_spike_counts, # 1 x trial x time x units
            coords={
                "frame_id": [frame_id],
                "trials": np.arange(stim_presentation_id.size)+1,
                "time_relative_to_stimulus_onset": time,
                "units": unit_id_multi_index
            },
            dims=[
                "frame_id",
                "trials",
                "time_relative_to_stimulus_onset",
                "units"
            ]
        )

        # Convert spike counts to firing rates: see cell 10 (cell 9 is basically the above function) of
        # https://github.com/jsiegle/AllenSDK/blob/opto-tutorial/doc_template/examples_root/
        # examples/nb/ecephys_optotagging.ipynb
        frs = spike_counts / TIME_RESOLUTION

        return frs

    def get_frs_across_sessions(self, stim_type="natural_scenes"):
        # Loop through each session ID
        frs_across_sessions = None
        for session_id in self.bo_sessions.index.values:
            print(f"Session: {session_id}")

            # Get data for session associated with current session_id and get the specimen
            # ID for the current session. This assumes that each session is associated with
            # one specimen.
            current_session = self.cache.get_session_data(session_id)
            current_specimen_id = self.bo_sessions.loc[session_id,:]["specimen_id"]
            assert isinstance(current_specimen_id, np.int64)

            # Check that there are 119 unique stimuli, including gray stimulus and are labelled
            # as -1,0,1,2,3,...,117
            ns_presentations_df = current_session.get_stimulus_table(stim_type)
            assert np.array_equal(np.unique(ns_presentations_df["frame"]), np.arange(-1, NUM_NS_STIM)), \
                    f"Session {session_id} only has the following frames: {np.unique(ns_presentations_df['frame'])}"

            # Gather units across desired visual areas. Each session has units in at least one
            # of these visual areas.
            units = current_session.units[current_session.units["ecephys_structure_acronym"].isin(VISUAL_AREAS)]

            # Loop through each natural scene frame in the current session
            frs_across_conditions = None
            for frame_id in np.arange(-1, NUM_NS_STIM):
                print(f"Frame {frame_id}")

                # Extract presentation IDs associated with frame_id
                cond_pres_ids = self.get_frame_presentation_ids(current_session, frame_id, stim_type)

                # Dimensions of spike_counts: ('stimulus_presentation_id', 'time_relative_to_stimulus_onset', 'unit_id')
                print(f"Getting spike counts for frame {frame_id}.")
                spike_counts = self.get_spike_counts_for_condition(current_session, cond_pres_ids, units)

                # Create new data structure which includes more metadata about each unit
                frs = self.add_metadata_to_spike_count_array(session_id, spike_counts, frame_id, current_specimen_id, units)

                # Concatenate across the per condition data array.
                if frs_across_conditions is None:
                    frs_across_conditions = frs
                else:
                    assert "frame_id" in frs.dims
                    frs_across_conditions = xr.concat([frs_across_conditions, frs], dim="frame_id")

            print(f"  {frs_across_conditions.shape}")

            # Dimensions: (num_conditions, num_trials, num_time, num_units)
            assert frs_across_conditions.shape[:3] == (NUM_NS_STIM_PLUS_GREY, NUM_TRIALS, NUM_TIMEPOINTS)

            # Now concatenate the across conditions data array along the units dimensions. We are
            # combining units (i.e. neurons) across specimens / sessions.
            if frs_across_sessions is None:
                frs_across_sessions = frs_across_conditions
            else:
                assert "units" in frs_across_conditions.dims
                frs_across_sessions = xr.concat([frs_across_sessions, frs_across_conditions], dim="units")

        # originally the data was images x trials x time x neurons, now we make it trials x images x time x neurons
        frs_across_sessions = frs_across_sessions.transpose('trials', 'frame_id',...)
        # drop grey stimulus (frame_id == -1)
        frs_across_sessions = frs_across_sessions.loc[dict(frame_id=slice(0, NUM_NS_STIM))]
        return frs_across_sessions

    def get_stimuli(self, frame_ids):
        """
        Get natural scene stimuli

        Inputs:
            frame_ids : (numpy.ndarray) array of frame IDs

        Outputs:
            stimuli : (numpy.ndarray) array of stimuli of dimensions (num_stimuli, height, width)
        """
        assert isinstance(frame_ids, np.ndarray)
        # Gray stimulus has a frame id of -1 and we exclude it in the neural data
        assert np.unique(frame_ids).size == NUM_NS_STIM # excluding gray stimulus

        stimuli = list()
        for i in range(NUM_NS_STIM):
            current_stim = self.cache.get_natural_scene_template(i)
            stimuli.append(current_stim)

        return np.array(stimuli)

    def add_split_half_reliabilities(self, neural_data, file_prefix="reliabilities"):
        """
        Add precomputed split-half reliabilities (num_time, num_neurons).
        You do not run this the first time the data is generated.

        Inputs:
            firing rates: (xarray) of neural responses of dimensions (num_trials, num_stimuli, num_time, num_neurons)

        Outputs:
            firing rates: (xarray) of neural responses of dimensions (num_trials, num_stimuli, num_time, num_neurons)
            but with multi-dimensional coordinates of split half reliabilities (mean and std across bootstraps)

        """

        neural_data_mod = []
        for v in VISUAL_AREAS:
            visual_area_data = neural_data.sel(visual_area = v)
            # the visual_area is deleted in visual_area_data from multi index
            # unintended behavior with multi indexes according to: https://github.com/pydata/xarray/issues/1408
            # (anyway, we add it back)
            unit_id_multi_index = pd.MultiIndex.from_arrays(
                [
                    visual_area_data.units.unit_id.values,
                    visual_area_data.units.specimen_id.values,
                    np.array([v]*len(visual_area_data.units)).astype('U'),
                    visual_area_data.units.image_selectivity_ns.values,
                    visual_area_data.units.run_pval_ns.values
                ],
                names=("unit_id", "specimen_id", "visual_area", "image_selectivity_ns", "run_pval_ns")
            )
            visual_area_data = visual_area_data.assign_coords(units=unit_id_multi_index)
            # now we add the reliabilities (mean and std across bootstraps)
            curr_area_rels_mean = [open_dataset(os.path.join(NEURAL_RESULTS_DIR, file_prefix + "_{}_t{}.pkl".format(v, t)))['r'] for t in range(NUM_TIMEPOINTS)]
            # time x num_neurons
            curr_area_rels_mean = xr.concat(curr_area_rels_mean, dim='time_relative_to_stimulus_onset')
            curr_area_rels_std = [open_dataset(os.path.join(NEURAL_RESULTS_DIR, file_prefix + "_{}_t{}.pkl".format(v, t)))['spread'] for t in range(NUM_TIMEPOINTS)]
            # time x num_neurons
            curr_area_rels_std = xr.concat(curr_area_rels_std, dim='time_relative_to_stimulus_onset')
            visual_area_data = visual_area_data.assign_coords(splithalf_r_mean=(('time_relative_to_stimulus_onset', 'units'), curr_area_rels_mean),
                                                              splithalf_r_std=(('time_relative_to_stimulus_onset', 'units'), curr_area_rels_std))

            neural_data_mod.append(visual_area_data)

        # concatenate across visual areas
        neural_data_mod = xr.concat(neural_data_mod, dim='units')

        return neural_data_mod

    def save_data(self, include_rels=True, save_path=None, verbose=True):
        if not os.path.isfile(NEUROPIX_DATA_PATH):
            frs_across_sessions = self.get_frs_across_sessions()
        else:
            print(f"Loading pre-packaged neuropixel data from {NEUROPIX_DATA_PATH}.")
            data = open_dataset(NEUROPIX_DATA_PATH)
            assert "neural_data" in data.keys()
            frs_across_sessions = data["neural_data"]

        if include_rels:
            frs_across_sessions = self.add_split_half_reliabilities(neural_data=frs_across_sessions)

        self.neural_data = frs_across_sessions
        assert "frame_id" in self.neural_data.dims
        self.stimuli = self.get_stimuli(self.neural_data["frame_id"].values)

        if verbose:
            print(f"Stimuli shape: {self.stimuli.shape} Neural data shape: {self.neural_data.shape}")

        self.packaged_data = {
            "stimuli": self.stimuli,
            "neural_data": self.neural_data
        }

        if save_path is None:
            if include_rels:
                self.save_data_path = NEUROPIX_DATA_PATH_WITH_RELS
            else:
                self.save_data_path = NEUROPIX_DATA_PATH
        else:
            self.save_data_path = save_path

        pickle.dump(self.packaged_data, open(self.save_data_path, "wb"), protocol=4)

def main():
    data_obj = NeuropixelsDataProcessor(MANIFEST_PATH)
    data_obj.save_data()

if __name__ == "__main__":
    main()


