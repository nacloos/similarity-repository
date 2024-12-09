import os
import pickle

import numpy as np

from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.default_dirs import NEURAL_RESULTS_DIR
from mouse_vision.core.names import filename_constructor
from mouse_vision.core.utils import dict_to_str
from mouse_vision.neural_data.model_data import generate_model_comparison_data
from mouse_vision.neural_mappers.utils import generate_train_test_img_splits
from mouse_vision.reliability.utils import package_reliability
from mouse_vision.reliability.metrics import noise_estimation

if __name__ == '__main__':
    '''We do not package these reliabilities with the model data since these metrics are subject to change as we undergo model comparisons,
    not to mention the model data will be subject to change too (which is why it is generated on the fly),
    whereas the full split half reliabilities were deliberately packaged with the original immutable data
    to enable subselection of self-consistent units in generate_model_data.'''

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="neuropixels", choices=["neuropixels", "calcium"])
    parser.add_argument("--num_train_test_splits", type=int, default=10)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--n_iter", type=int, default=900)
    parser.add_argument("--n_jobs", type=int, default=5)
    parser.add_argument("--splithalf_r_thresh", type=float, default=None)
    args = parser.parse_args()

    metric = args.metric if args.metric is not None else "pearsonr"
    data_kwargs = {"center_trials": False, "dataset_name": args.dataset}
    if args.splithalf_r_thresh is not None:
        data_kwargs["splithalf_r_thresh"] = args.splithalf_r_thresh

    model_data = generate_model_comparison_data(**data_kwargs)

    for v in VISUAL_AREAS:
        curr_area_resp = model_data[v]
        train_test_splits = generate_train_test_img_splits(num_splits=args.num_train_test_splits,
                                                           train_frac=args.train_frac)

        results_agg = []
        for curr_sp in train_test_splits:
            curr_area_resp_testset = curr_area_resp.isel(frame_id=curr_sp['test'])
            curr_results = noise_estimation(
                curr_area_resp_testset,
                metric=metric,
                mode="spearman_brown_split_half",
                center=np.nanmean, # since some trials can be nan
                summary_center='raw',
                sync=True,
                parallelize_per_target_unit=False if metric == "rsa" else True,
                n_iter=args.n_iter,
                n_jobs=args.n_jobs
            )
            results_agg.append(curr_results)
        results_agg = np.concatenate(results_agg, axis=0)
        assert(results_agg.ndim < 3) # flattened train/test splits and bootstrap trials x units
        print(f"Computed results for area {v} of shape {results_agg.shape}")
        # average across these like we do with interanimal consistencies
        r = np.mean(results_agg, axis=0)
        s = np.std(results_agg, axis=0)
        if metric != "rsa":
            # package with unit info
            r = package_reliability(r, unit_index=curr_area_resp_testset.units)
            s = package_reliability(s, unit_index=curr_area_resp_testset.units)
        reliabilities = {"r": r, "spread": s}
        if metric == "rsa":
            # since there are no units to median over in this case we return the entire 1D array
            reliabilities["full"] = results_agg
        filename = filename_constructor(num_train_test_splits=args.num_train_test_splits,
                                        train_frac=args.train_frac,
                                        metric_name=args.metric,
                                        n_iter=args.n_iter,
                                        splithalf_r_thresh=args.splithalf_r_thresh,
                                        names='model_comparison_reliabilities_' + dict_to_str({'dataset_name': args.dataset, 'visual_area': v}))
        pickle.dump(
            reliabilities,
            open(
                os.path.join(
                    NEURAL_RESULTS_DIR,
                    filename
                ),
                "wb"
            )
        )
