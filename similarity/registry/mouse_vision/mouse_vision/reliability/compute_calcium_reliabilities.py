import os
import pickle
import numpy as np

from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.default_dirs import CALCIUM_DATA_PATH, NEURAL_RESULTS_DIR
from mouse_vision.core.utils import open_dataset
from mouse_vision.reliability.utils import package_reliability
from mouse_vision.reliability.metrics import noise_estimation

def compute_reliability(visual_area):
    d = open_dataset(CALCIUM_DATA_PATH)
    neural_resps = d["neural_data"]
    # trials x images x neurons as noise_estimation expects
    curr_area_resp = neural_resps.sel(area=visual_area)

    r, s = noise_estimation(
        curr_area_resp,
        metric="pearsonr",
        mode="spearman_brown_split_half",
        center=np.nanmean, # since some trials can be nan
        summary_center=np.mean,
        summary_spread=np.std,
        sync=True,
        n_jobs=5
    )

    r_xarray = package_reliability(r, unit_index=curr_area_resp.units)
    s_xarray = package_reliability(s, unit_index=curr_area_resp.units)

    reliabilities = {'r': r_xarray, "spread": s_xarray}
    pickle.dump(
        reliabilities,
        open(
            os.path.join(
                NEURAL_RESULTS_DIR,
                "ophys_data/reliabilities_{}.pkl".format(visual_area)
            ),
            "wb"
        )
    )

if __name__ == '__main__':
    import sys
    for vis_area in VISUAL_AREAS[::-1]:
        sys.stdout.write(f"Doing {vis_area}.\n")
        compute_reliability(vis_area)
        sys.stdout.write(f"Done {vis_area}.\n")
        sys.stdout.flush()


