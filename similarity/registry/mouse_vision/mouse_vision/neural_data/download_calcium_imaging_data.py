import numpy as np
import pandas as pd

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.natural_scenes import NaturalScenes

import allensdk.brain_observatory.stimulus_info as stim_info

# http://alleninstitute.github.io/AllenSDK/_modules/allensdk/core/brain_observatory_cache.html

boc = BrainObservatoryCache(manifest_file="/mnt/fs5/nclkong/allen_inst/ophys_data/manifest.json")

cells = boc.get_cell_specimens()
cells = pd.DataFrame.from_records(cells)
unique_specimens = np.unique(cells["specimen_id"])
for specimen_id in unique_specimens:
    print(f"Specimen ID: {specimen_id}")
    # Get all cell specimen ids for natural scene experiments
    cells_for_specimen = cells[cells["specimen_id"] == specimen_id]
    
    # Each cell has unique cell specimen id
    assert np.unique(cells_for_specimen["cell_specimen_id"]).size == cells_for_specimen.shape[0]
    
    cells_for_specimen_ids = cells_for_specimen["cell_specimen_id"].values
    
    # Get experiments
    exps = boc.get_ophys_experiments(cell_specimen_ids=cells_for_specimen_ids, 
                                     stimuli=[stim_info.NATURAL_SCENES])
    
    print(len(exps))
    for exp in exps:
        exp_id = exp["id"]
        print(f"  Downloading data for experiment ID: {exp_id}")
        exp_data = boc.get_ophys_experiment_data(exp_id)

