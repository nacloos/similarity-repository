import os
import pickle
import numpy as np

from mouse_vision.core.default_dirs import NEURAL_DATA_DIR, NEUROPIX_DATA_PATH, NEURAL_RESULTS_DIR
from mouse_vision.core.utils import open_dataset, get_params_from_workernum
from mouse_vision.core.constants import VISUAL_AREAS, TIME_BINS, NUM_TIMEPOINTS
from mouse_vision.reliability.utils import package_temporal_reliability
from mouse_vision.reliability.metrics import noise_estimation

def build_param_lookup(visual_areas=VISUAL_AREAS,
                       timepoints=np.arange(0, NUM_TIMEPOINTS)):
    # build param lookup
    param_lookup = {}
    key = 0
    for v in visual_areas:
        for t_idx in timepoints:
            param_lookup[str(key)] = {'visual_area': v,
                                      't_idx': t_idx}
            key += 1
    return param_lookup

def compute_reliability(visual_area, t_idx):
    d = open_dataset(NEUROPIX_DATA_PATH)
    neural_resps = d['neural_data']
    # trials x images x time x neurons
    curr_area_resps = neural_resps.sel(visual_area=visual_area)
    # trials x images x neurons as noise_estimation expects
    curr_time_area_resp = curr_area_resps[:, :, t_idx, :]

    r, s = noise_estimation(curr_time_area_resp,
                        metric='pearsonr',
                        mode='spearman_brown_split_half',
                        center=np.nanmean, # since some trials can be nan
                        summary_center=np.mean,
                        summary_spread=np.std,
                        sync=True)

    r_xarray = package_temporal_reliability(r,
                                   time_index=[curr_time_area_resp.time_relative_to_stimulus_onset],
                                   unit_index=curr_time_area_resp.units)
    s_xarray = package_temporal_reliability(s,
                                   time_index=[curr_time_area_resp.time_relative_to_stimulus_onset],
                                   unit_index=curr_time_area_resp.units)

    reliabilities = {'r': r_xarray, 'spread': s_xarray}
    pickle.dump(reliabilities, open(os.path.join(NEURAL_RESULTS_DIR, 'reliabilities_{}_t{}.pkl'.format(visual_area, t_idx))))

if __name__ == '__main__':
    print('Looking up params')
    param_lookup = build_param_lookup()
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print('Curr params', curr_params)
    compute_reliability(**curr_params)

