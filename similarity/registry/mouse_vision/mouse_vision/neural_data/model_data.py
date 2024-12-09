import itertools

import numpy as np
import xarray as xr

from collections import OrderedDict

from mouse_vision.core.utils import open_dataset
from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.dataloader_utils import duplicate_channels
from mouse_vision.reliability.spec_map_utils import generate_spec_map
from mouse_vision.core.default_dirs import \
    NEUROPIX_DATA_PATH_WITH_RELS, \
    CALCIUM_DATA_PATH_WITH_RELS

def check_resp_dims(resp,
                    center_trials=True):
    if center_trials:
        # nanmean across trials, since some trials can be nan
        resp = resp.mean(dim="trials", skipna=True)
        # images x neurons
        assert(resp.ndim == 2)
        assert(resp.dims[0] == "frame_id")
        assert(resp.dims[1] == "units")
    else:
        assert(resp.ndim == 3)
        assert(resp.dims[0] == "trials")
        assert(resp.dims[1] == "frame_id")
        assert(resp.dims[2] == "units")
    return resp

def generate_model_comparison_data(center_trials=True,
                                   return_stimuli=False,
                                   dataset_name="neuropixels",
                                   separate_by_animal=False,
                                   **kwargs):
    '''
    We do not explicitly package this data since the criteria by which we subselect
    units for model comparisons can be subject to change. Enables them to be altered
    flexibly "on the fly" from the original, immutable packaged data.

    Inputs:
        center_trials  : (boolean) whether to return data that is averaged across trials
        return_stimuli : (boolean) whether to return stimulus set with duplicated channels
                         (i.e., grayscale to RGB)
        dataset_name   : (string) one of ["neuropixels", "calcium"]
        separate_by_animal: (boolean) return dictionary of responses per animal if True, else return array aggregated across animals
    '''

    # Return stimuli array?
    if return_stimuli:
        if dataset_name == "neuropixels":
            dataset = open_dataset(NEUROPIX_DATA_PATH_WITH_RELS)
        elif dataset_name == "calcium":
            dataset = open_dataset(CALCIUM_DATA_PATH_WITH_RELS)

        # Duplicate gray channels to RGB
        assert "stimuli" in dataset.keys()
        stimuli = duplicate_channels(dataset["stimuli"])
        del dataset

    spec_map = generate_spec_map(dataset_name=dataset_name, **kwargs)

    # We keep the data separated by visual area for convenience/speed since we will
    # in most cases be fitting models separately per visual area
    model_data = OrderedDict()
    for v in VISUAL_AREAS:
        specimens = []
        if separate_by_animal:
            curr_area_resp = OrderedDict()
        else:
            curr_area_resp = []
            specimen_id_meta = []

        for s in spec_map[v].keys():
            if isinstance(s, np.int64):
                specimens.append(s)
                if separate_by_animal:
                    curr_area_resp[s] = check_resp_dims(resp=spec_map[v][s],
                                                        center_trials=center_trials)
                else:
                    curr_spec_resp = spec_map[v][s]
                    curr_area_resp.append(curr_spec_resp)
                    specimen_id_meta.extend([s]*len(curr_spec_resp.units))

        specimen_pairs = list(itertools.combinations(specimens, r=2))

        # Ensures that specimens and the order they were collected in spec map has been preserved
        assert(np.array_equal(np.array(specimen_pairs), np.array(spec_map[v]["specimen_pairs"])) is True)
        if not separate_by_animal:
            curr_area_resp = xr.concat(curr_area_resp, dim="units")

            # Add back specimen id and visual area information as a coord to avoid reindexing unit
            # dimension and making assumptions about its contents, causing potential information loss
            specimen_id_meta = np.array(specimen_id_meta)
            visual_area_meta = np.array([v]*len(curr_area_resp.units)).astype('U')
            curr_area_resp = curr_area_resp.assign_coords(specimen_id=("units", specimen_id_meta),
                                                          visual_area=("units", visual_area_meta))


            curr_area_resp = check_resp_dims(resp=curr_area_resp,
                                             center_trials=center_trials)

        model_data[v] = curr_area_resp

    if return_stimuli:
        return model_data, stimuli
    else:
        return model_data

