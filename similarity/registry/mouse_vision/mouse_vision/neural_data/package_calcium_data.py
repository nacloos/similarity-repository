import os
import sys
import pickle

import numpy as np
import pandas as pd
import xarray as xr

import allensdk.brain_observatory.stimulus_info as stim_info

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.natural_scenes import NaturalScenes

from mouse_vision.core.utils import open_dataset
from mouse_vision.core.default_dirs import \
    NEURAL_RESULTS_DIR, \
    CALCIUM_DATA_PATH, \
    CALCIUM_DATA_PATH_WITH_RELS, \
    CALCIUM_MANIFEST_PATH
from mouse_vision.core.constants import \
    VISUAL_AREAS, \
    NUM_TRIALS, \
    NUM_NS_STIM, \
    NUM_NS_STIM_PLUS_GREY

class CalciumImagingDataProcessor():
    def __init__(self, manifest_path):
        self.cache = BrainObservatoryCache(manifest_file=manifest_path)
        self.cells = pd.DataFrame.from_records(self.cache.get_cell_specimens())
        self.unique_specimens = np.unique(self.cells["specimen_id"])
        self.stimuli = None

    def _process_data_array(self, data_array, frames):
        """
        Organizes the trial x cells array into 50 trials x 119 stimuli x cells array.
        Outputs an xarray with necessary metadata.

        Inputs:
            data_array : (num_trials, num_cells)
            frames     : (num_trials,) each entry tells us which stimulus is associated
                         with the data

        Outputs:
            image_responses : (numpy.ndarray) array of responses (50, n_stim, n_cells)
        """
        # Frames are ordered as: -1,0,1,2,...,NUM_NS_STIM-1. -1 is the gray stimulus
        assert np.unique(frames).size == NUM_NS_STIM_PLUS_GREY and \
                np.min(frames) == -1 and \
                np.max(frames) == NUM_NS_STIM - 1

        # We index starting from 0 because we drop the gray stimulus, which has a
        # frame id of -1.
        image_responses = []
        frame_ids = []
        for i in range(0, NUM_NS_STIM):
            idx = (frames == i)
            frame_ids.append(i)
            if int(idx.sum()) < NUM_TRIALS:
                print(f"  [WARNING] Less than {NUM_TRIALS} trials ({idx.sum()} trials found).")
                n_missing = int(NUM_TRIALS - idx.size)
                nan_array = np.full((n_missing,n_cells), np.NaN)

                responses = data_array[idx] # (n_trial x n_cells)
                concat = np.concatenate((responses, nan_array), axis=0)
                image_responses.append(concat)
            elif int(idx.sum()) > NUM_TRIALS:
                raise ValueError(f"This should not occur. Number of trials found: {int(idx.size)}.")
            else:
                assert int(idx.sum()) == NUM_TRIALS
                image_responses.append(data_array[idx])

        image_responses = np.array(image_responses) # (n_stim, 50, n_cells)
        image_responses = np.transpose(image_responses, (1,0,2)) # (50, n_stim, n_cells)

        return image_responses, np.array(frame_ids)

    def _get_cell_metadata(self, cell_ids):
        assert "cell_specimen_id" in self.cells.columns
        # We need to do this to maintain the ordering in cell_ids
        # See: https://stackoverflow.com/questions/56658723/how-to-maintain-order-when-selecting-rows-in-pandas-dataframe
        cells2 = self.cells.set_index("cell_specimen_id") # TODO: Might want to put this in the __init__()

        # cell_data index names line up with cell_ids
        cell_data = cells2.loc[cell_ids]
        assert np.array_equal(cell_ids, cell_data.index.values)

        desired_metadata = ["area", "image_sel_ns", "p_ns", "p_run_mod_ns", "peak_dff_ns", \
                            "pref_image_ns", "reliability_ns", "run_mod_ns", "specimen_id", \
                            "time_to_peak_ns"]

        metadata_dict = {}
        for name in desired_metadata:
            assert name in cell_data.columns, "{name} not in cells data frame."
            metadata_dict[name] = cell_data[name].values

        assert "cell_specimen_id" == cell_data.index.name
        metadata_dict["cell_specimen_id"] = cell_data.index.values
        assert np.array_equal(cell_ids, cell_data.index.values)

        return metadata_dict

    def _process_exp_data(self, exp_id, specimen_id):
        """
        Processes experiment data associated with exp_id into an xarray.

        Inputs:
            exp_id      : (int)
            specimen_id : (int)

        Outputs:
            data_array : (xarray.DataArray)
        """
        print(f"  Processing experiment ID: {exp_id}...")
        exp_data = self.cache.get_ophys_experiment_data(exp_id)

        cell_ids = exp_data.get_cell_specimen_ids()
        cell_metadata = self._get_cell_metadata(cell_ids)

        ns = NaturalScenes(exp_data)
        data_array = ns.mean_sweep_response.values[:,:ns.numbercells]
        frames = exp_data.get_stimulus_table(stim_info.NATURAL_SCENES)["frame"].values

        data_array, frame_ids = self._process_data_array(data_array, frames)
        assert data_array.shape[0] == NUM_TRIALS, f"Data array shape: {data_array.shape}" # 50 trials
        # It is "-1" here since we dropped the gray stimulus
        assert data_array.shape[1] == frame_ids.size == np.unique(frames).size - 1, \
                    f"Data array shape: {data_array.shape}"
        assert data_array.shape[2] == ns.numbercells == cell_ids.size, \
                    f"Data array shape: {data_array.shape}, number of cells: {ns.numbercells}"

        def _construct_metadata_dict(metadata, num_cells):
            data = []
            names = []
            for k, v in metadata.items():
                names.append(k)
                data.append(v)
                assert v.size == num_cells, f"Mismatch: {v.size}, {num_cells}"
            return data, names

        # Construct metadata array for cells
        metadata, dim_names = _construct_metadata_dict(cell_metadata, ns.numbercells)
        unit_id_multi_index = pd.MultiIndex.from_arrays(metadata, names=tuple(dim_names))

        # Construct xarray data array
        data_array = xr.DataArray(
            data_array, # 50 x stim x cells
            coords={
                "trials": np.arange(NUM_TRIALS)+1,
                "frame_id": frame_ids,
                "units": unit_id_multi_index
            },
            dims=["trials", "frame_id", "units"]
        )

        # Get stimuli
        scenes = exp_data.get_stimulus_template(stim_info.NATURAL_SCENES)
        assert scenes.shape[0] == data_array.shape[1], \
            f"Different number of stimuli, {data_array.shape} vs. {scenes.shape}"
        if self.stimuli is None:
            self.stimuli = scenes
        else:
            assert np.array_equal(self.stimuli, scenes), "Stimuli are not same across experiments."

        return data_array

    def _process_specimen_data(self, specimen_id):
        print(f"Processing specimen {specimen_id}...")

        cells_for_specimen = self.cells[self.cells["specimen_id"] == specimen_id]
        assert np.unique(cells_for_specimen["cell_specimen_id"]).size == cells_for_specimen.shape[0]
        cells_for_specimen_ids = cells_for_specimen["cell_specimen_id"].values

        exps = self.cache.get_ophys_experiments(
            cell_specimen_ids=cells_for_specimen_ids,
            stimuli=[stim_info.NATURAL_SCENES]
        )
        print(f"  Specimen {specimen_id} has {len(exps)} experiments.")

        responses_across_exps = None
        for exp in exps:
            exp_id = exp["id"]
            # data_array: (n_trial, n_stim, n_cells)
            data_array = self._process_exp_data(exp_id, specimen_id)
            assert "units" in data_array.dims

            if responses_across_exps is None:
                responses_across_exps = data_array
            else:
                responses_across_exps = xr.concat([responses_across_exps, data_array], dim="units")
        return responses_across_exps

    def process_data(self):
        responses_across_specimens = None
        for i, specimen_id in enumerate(self.unique_specimens):

            responses = self._process_specimen_data(specimen_id)
            if responses is None:
                continue

            print(f"Specimen ID {specimen_id}, response shape: {responses.shape}")
            sys.stdout.flush()

            assert "units" in responses.dims

            if responses_across_specimens is None:
                responses_across_specimens = responses
            else:
                responses_across_specimens = xr.concat([responses_across_specimens, responses], dim="units")

        return responses_across_specimens

    def add_split_half_reliabilities(self, neural_data, file_prefix="reliabilities"):
        """
        Add precomputed split-half reliabilities (num_neurons,).
        You do not run this the first time the data is generated.

        Inputs:
            neural_data: (xarray) of neural responses of dimensions (num_trials, num_stimuli, num_neurons)

        Outputs:
            neural_data_mod: (xarray) of neural responses of dimensions (num_trials, num_stimuli, num_neurons)
                             but with multi-dimensional coordinates of split half reliabilities
                             (mean and std across bootstraps)
        """
        vis_area_mod = []
        for v in VISUAL_AREAS:
            print(f"Doing visual area {v}")
            reliabilities_base_dir = os.path.join(NEURAL_RESULTS_DIR, "ophys_data/")
            reliabilities_file = reliabilities_base_dir + f"{file_prefix}_{v}.pkl"
            rels = open_dataset(reliabilities_file)
            r = rels['r'].values
            s = rels["spread"].values

            vis_area_dat = neural_data.sel(area=v)
            # Make sure cells in reliabilities results line up with cells in the original calcium
            # packaged data.
            assert np.array_equal(vis_area_dat.cell_specimen_id.values, rels['r'].units.cell_specimen_id.values)

            # Add back area coordinate
            area = np.array([v]*len(vis_area_dat.units)).astype('U')
            additional_data = vis_area_dat.units.to_index().to_frame()
            additional_data["visual_area"] = area
            additional_data["reliability_mean"] = r
            additional_data["reliability_std"] = s
            mi = pd.MultiIndex.from_frame(additional_data)

            vis_area_dat = vis_area_dat.assign_coords(units=mi)
            vis_area_mod.append(vis_area_dat)

        neural_data_mod = xr.concat(vis_area_mod, dim="units")
        return neural_data_mod

    def save_data(self, include_rels=True, save_path=None, verbose=True):
        if not os.path.isfile(CALCIUM_DATA_PATH):
            self.neural_data = self.process_data()
        else:
            print(f"Opening pre-packaged calcium data at {CALCIUM_DATA_PATH}.")
            data = open_dataset(CALCIUM_DATA_PATH)
            assert "neural_data" in data.keys()
            assert "stimuli" in data.keys()
            self.neural_data = data["neural_data"]
            self.stimuli = data["stimuli"]

        if include_rels:
            self.neural_data = self.add_split_half_reliabilities(self.neural_data)

        assert self.stimuli is not None, "Stimuli cannot be nonexistent!"
        if verbose:
            print(f"Stimuli shape: {self.stimuli.shape} Neural data shape: {self.neural_data.shape}")

        self.packaged_data = {
            "stimuli": self.stimuli,
            "neural_data": self.neural_data
        }

        if save_path is None:
            if include_rels:
                self.save_data_path = CALCIUM_DATA_PATH_WITH_RELS
            else:
                self.save_data_path = CALCIUM_DATA_PATH
        else:
            self.save_data_path = save_path

        pickle.dump(self.packaged_data, open(self.save_data_path, "wb"), protocol=4)

def main():
    c = CalciumImagingDataProcessor(CALCIUM_MANIFEST_PATH)
    c.save_data()


if __name__ == "__main__":
    main()



