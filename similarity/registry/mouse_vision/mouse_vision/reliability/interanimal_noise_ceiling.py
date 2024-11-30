import os
import copy
import pickle
import itertools

import numpy as np
import xarray as xr
from mouse_vision.core.names import filename_constructor
from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.default_dirs import NEURAL_RESULTS_DIR, INTERAN_TR_DIR
from mouse_vision.core.utils import get_params_from_workernum
from mouse_vision.neural_mappers.utils import generate_train_test_img_splits
from mouse_vision.reliability.spec_map_utils import generate_spec_map
from mouse_vision.neural_mappers.map_param_utils import check_source_map_kw, generate_map_param_grid
from mouse_vision.reliability.metrics import noise_estimation

def build_param_lookup(source_map_kwargs=None,
                       map_type=None,
                       map_param_grid_idxs=None,
                       dataset_name='neuropixels',
                       equalize_source_units=True,
                       visual_areas=VISUAL_AREAS,
                       specimen_pairs=None,
                       fit_per_target_unit=False,
                       parallelize_per_target_unit=False,
                       metric=None,
                       num_train_test_splits=10,
                       n_ss_iter=20,
                       n_iter=200,
                       splithalf_r_thresh=None,
                       train_frac=0.5,
                       mode='pairwise',
                       correction='spearman_brown_split_half',
                       shorten_name=False,
                       report_train=False,
                       n_jobs=5):

    if not fit_per_target_unit:
        # cannot parallelize/split jobs by target unit if fitting map across all target units simultaneously
        # instead of per target unit separately
        assert(parallelize_per_target_unit is False)

    # convert visual areas to list if need be
    if not isinstance(visual_areas, list):
        visual_areas = [visual_areas]

    # generate basic information per specimen and across them
    data_kwargs = {"dataset_name": dataset_name, "equalize_source_units": equalize_source_units}
    if splithalf_r_thresh is not None:
        data_kwargs["splithalf_r_thresh"] = splithalf_r_thresh

    visual_area_spec_map = generate_spec_map(**data_kwargs)

    # use automatically generated source map kwargs grid if not provided by the user
    if source_map_kwargs is None:
        assert(map_type is not None)
        source_map_kwargs = generate_map_param_grid(map_type=map_type,
                                                    map_param_grid_idxs=map_param_grid_idxs,
                                                    visual_areas=visual_areas,
                                                    visual_area_spec_map=visual_area_spec_map,
                                                    fit_per_target_unit=fit_per_target_unit)

    # integrity checks
    source_map_kwargs = check_source_map_kw(source_map_kwargs=source_map_kwargs,
                                            visual_areas=visual_areas)

    # build param lookup
    param_lookup = {}
    key = 0
    for v in visual_areas:
        curr_num_source_units = visual_area_spec_map[v].get('num_source_units', None)
        curr_n_ss_iter = None if curr_num_source_units is None else n_ss_iter
        if mode == 'pairwise':
            if specimen_pairs is None:
                curr_spec_pairs = visual_area_spec_map[v]['specimen_pairs']
            else:
                curr_spec_pairs = specimen_pairs[v]

            for map_kw in source_map_kwargs[v]:
                for spec_pair in curr_spec_pairs:
                    assert(len(spec_pair) == 2)
                    source = spec_pair[0]
                    target = spec_pair[1]
                    pair1 = {
                        'names': {
                            'dataset_name': dataset_name, 'visual_area': v, 'spec_pair': spec_pair
                        },
                        'source_N': visual_area_spec_map[v][source],
                        'target_N': visual_area_spec_map[v][target],
                        'num_source_units': curr_num_source_units,
                        'source_map_kwargs': map_kw,
                        'parallelize_per_target_unit': parallelize_per_target_unit,
                        'n_jobs': n_jobs,
                        'metric': metric,
                        'num_train_test_splits': num_train_test_splits,
                        'n_ss_iter': curr_n_ss_iter,
                        'n_iter': n_iter,
                        'train_frac': train_frac,
                        'correction': correction,
                        'report_train': report_train,
                        'shorten_name': shorten_name,
                        'splithalf_r_thresh': splithalf_r_thresh,
                    }

                    # Flip source/target since we want *permutations*
                    pair2 = copy.deepcopy(pair1)
                    pair2['names']['spec_pair'] = spec_pair[::-1]
                    pair2['source_N'] = visual_area_spec_map[v][target]
                    pair2['target_N'] = visual_area_spec_map[v][source]

                    if dataset_name != "neuropixels":
                        # If calcium, then each job will run source->target *and* target->source
                        param_lookup[str(key)] = [pair1, pair2]
                        key += 1
                    else:
                        param_lookup[str(key)] = [pair1]
                        key += 1
                        param_lookup[str(key)] = [pair2]
                        key += 1

        elif mode == 'holdout':
            curr_specimens = [k for k in visual_area_spec_map[v].keys() if isinstance(k, np.int64)]
            assert(len(list(itertools.combinations(curr_specimens, r=2))) == len(visual_area_spec_map[v]['specimen_pairs']))
            for map_kw in source_map_kwargs[v]:
                for curr_spec in curr_specimens:
                    spec_pair = ('holdout', curr_spec)
                    source_N = xr.concat([visual_area_spec_map[v][source_spec] for source_spec in curr_specimens if source_spec != curr_spec],
                                        dim="units")
                    target_N = visual_area_spec_map[v][curr_spec]
                    cons_kwargs = {
                        'names': {
                            'dataset_name': dataset_name, 'visual_area': v, 'spec_pair': spec_pair
                        },
                        'source_N': source_N,
                        'target_N': target_N,
                        'num_source_units': curr_num_source_units,
                        'source_map_kwargs': map_kw,
                        'parallelize_per_target_unit': parallelize_per_target_unit,
                        'n_jobs': n_jobs,
                        'metric': metric,
                        'num_train_test_splits': num_train_test_splits,
                        'n_ss_iter': curr_n_ss_iter,
                        'n_iter': n_iter,
                        'train_frac': train_frac,
                        'correction': correction,
                        'report_train': report_train,
                        'shorten_name': shorten_name,
                        'splithalf_r_thresh': splithalf_r_thresh,
                    }

                    param_lookup[str(key)] = [cons_kwargs]
                    key += 1
        else:
            raise ValueError

    return param_lookup

def compute_interanimal_consistencies(source_N,
                                      target_N,
                                      source_map_kwargs,
                                      num_source_units=None,
                                      num_train_test_splits=10,
                                      n_ss_iter=20,
                                      n_iter=200,
                                      train_frac=0.5,
                                      metric=None,
                                      names=None,
                                      parallelize_per_target_unit=False,
                                      correction='spearman_brown_split_half',
                                      n_jobs=5,
                                      report_train=False,
                                      shorten_name=False,
                                      splithalf_r_thresh=None):

    if report_train:
        agg_results_train = []

    agg_results = []
    train_test_splits = generate_train_test_img_splits(num_splits=num_train_test_splits,
                                                       train_frac=train_frac)
    for curr_sp in train_test_splits:
        curr_results = noise_estimation(target_N=target_N,
                                        source_N=source_N,
                                        source_map_kwargs=source_map_kwargs,
                                        parallelize_per_target_unit=parallelize_per_target_unit,
                                        train_img_idx=curr_sp['train'], test_img_idx=curr_sp['test'],
                                        num_source_units=num_source_units,
                                        metric=metric if metric is not None else 'pearsonr',
                                        mode=correction,
                                        center=np.nanmean,
                                        summary_center='raw',
                                        sync=True,
                                        n_ss_iter=n_ss_iter,
                                        n_iter=n_iter,
                                        report_train=report_train,
                                        n_jobs=n_jobs)
        if report_train:
            curr_results_test, curr_results_train = curr_results
            curr_results_test = np.expand_dims(curr_results_test, axis=0)
            curr_results_train = np.expand_dims(curr_results_train, axis=0)
            agg_results.append(curr_results_test)
            agg_results_train.append(curr_results_train)
        else:
            curr_results = np.expand_dims(curr_results, axis=0)
            agg_results.append(curr_results)

    # (num_train_test_splits, num_bs_trials, num_target_units)
    if report_train:
        agg_results_train = np.concatenate(agg_results_train, axis=0)

    agg_results = np.concatenate(agg_results, axis=0)

    filename = filename_constructor(source_map_kwargs=source_map_kwargs,
                                    num_source_units=num_source_units,
                                    num_train_test_splits=num_train_test_splits,
                                    n_ss_iter=n_ss_iter,
                                    n_iter=n_iter,
                                    train_frac=train_frac,
                                    fit_per_target_unit=parallelize_per_target_unit,
                                    metric_name=metric,
                                    correction=correction,
                                    names=names,
                                    shorten_name=shorten_name,
                                    splithalf_r_thresh=splithalf_r_thresh)
    if report_train:
        filename = os.path.join(INTERAN_TR_DIR, filename)
        pickle.dump({"train": agg_results_train, "test": agg_results}, open(filename, "wb"), protocol=4)
    else:
        filename = os.path.join(NEURAL_RESULTS_DIR, filename)
        pickle.dump(agg_results, open(filename, "wb"), protocol=4)

if __name__ == '__main__':
    # TODO: Use generate_map_param_grid for correlation neuralmapper similar to PLS

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="neuropixels", choices=["neuropixels", "calcium"])
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--map_type", type=str, default=None, required=True)
    parser.add_argument("--map_param_grid_idx", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=5)
    parser.add_argument("--pls_n_components", type=int, default=None)
    parser.add_argument("--pls_fit_target_unit_population", type=bool, default=False)
    parser.add_argument("--parallelize_per_target_unit", type=bool, default=False)
    parser.add_argument("--num_train_test_splits", type=int, default=10)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--n_iter", type=int, default=200)
    parser.add_argument("--n_ss_iter", type=int, default=20)
    parser.add_argument("--keep_source_unchanged", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="pairwise", choices=["pairwise", "holdout"])
    parser.add_argument("--correction", type=str, default="spearman_brown_split_half")
    parser.add_argument("--report_train", type=bool, default=False)
    parser.add_argument("--shorten_name", type=bool, default=False)
    parser.add_argument("--splithalf_r_thresh", type=float, default=None)
    args = parser.parse_args()

    equalize_source_units = False if args.keep_source_unchanged else True
    print("Looking up params...")
    if args.map_type == "identity":
        param_lookup = build_param_lookup(
            map_type="identity",
            metric="rsa", # only rsa makes sense for the identity map since the number of neurons between source and target can be unequal
            dataset_name=args.dataset,
            n_jobs=args.n_jobs,
            num_train_test_splits=args.num_train_test_splits,
            train_frac=args.train_frac,
            n_iter=args.n_iter,
            n_ss_iter=args.n_ss_iter,
            equalize_source_units=equalize_source_units,
            mode=args.mode,
            correction=args.correction,
            report_train=False, # no train set in RSA
            shorten_name=args.shorten_name,
            splithalf_r_thresh=args.splithalf_r_thresh,
        )
    elif args.map_type in ["corr", "ridge", "lasso", "elasticnet"]:
        if args.map_type == "corr":
            source_map_kwargs = {
                'map_type': 'corr', 'map_kwargs': {'identity': True, 'percentile': 100}
            }
        elif args.map_type == "ridge":
            source_map_kwargs = {
                'map_type': 'corr', 'map_kwargs': {'regression_type': "Ridge", 'identity': False, 'percentile': 0}
            }
        elif args.map_type == "lasso":
            source_map_kwargs = {
                'map_type': 'corr', 'map_kwargs': {'regression_type': "Lasso", 'identity': False, 'percentile': 0}
            }
        elif args.map_type == "elasticnet":
            source_map_kwargs = {
                'map_type': 'corr', 'map_kwargs': {'regression_type': "ElasticNet", 'identity': False, 'percentile': 0}
            }
        else:
            raise ValueError

        param_lookup = build_param_lookup(
            source_map_kwargs=source_map_kwargs,
            metric=args.metric,
            dataset_name=args.dataset,
            n_jobs=args.n_jobs,
            num_train_test_splits=args.num_train_test_splits,
            train_frac=args.train_frac,
            n_iter=args.n_iter,
            n_ss_iter=args.n_ss_iter,
            equalize_source_units=equalize_source_units,
            mode=args.mode,
            correction=args.correction,
            report_train=args.report_train,
            shorten_name=args.shorten_name,
            splithalf_r_thresh=args.splithalf_r_thresh,
        )
    elif args.map_type == "pls":
        pls_fit_per_target_unit = False if args.pls_fit_target_unit_population else True
        if pls_fit_per_target_unit:
            assert(args.parallelize_per_target_unit is False)
        if args.pls_n_components is not None:
            source_map_kwargs = {
                'map_type': 'pls', 'map_kwargs': {'n_components': args.pls_n_components, 'fit_per_target_unit': pls_fit_per_target_unit}
            }
        else:
            source_map_kwargs = None
        param_lookup = build_param_lookup(
            map_type="pls",
            source_map_kwargs=source_map_kwargs,
            metric=args.metric,
            dataset_name=args.dataset,
            map_param_grid_idxs=args.map_param_grid_idx,
            n_jobs=args.n_jobs,
            fit_per_target_unit=pls_fit_per_target_unit,
            parallelize_per_target_unit=args.parallelize_per_target_unit,
            num_train_test_splits=args.num_train_test_splits,
            train_frac=args.train_frac,
            n_iter=args.n_iter,
            n_ss_iter=args.n_ss_iter,
            equalize_source_units=equalize_source_units,
            mode=args.mode,
            correction=args.correction,
            report_train=args.report_train,
            shorten_name=args.shorten_name,
            splithalf_r_thresh=args.splithalf_r_thresh,
        )
    else:
        raise ValueError(f"{args.map_type} for {dataset} not implemented yet.")

    print("TOTAL NUMBER OF JOBS: {}".format(len(list(param_lookup.keys()))))
    curr_params = get_params_from_workernum(worker_num=os.environ.get('SLURM_ARRAY_TASK_ID'), param_lookup=param_lookup)
    print("Current params", curr_params)
    for curr_param in curr_params:
        compute_interanimal_consistencies(**curr_param)


