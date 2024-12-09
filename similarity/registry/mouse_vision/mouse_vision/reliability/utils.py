import numpy as np
import xarray as xr
import os
import pickle
import itertools
from mouse_vision.core.utils import open_dataset, dict_to_str
from mouse_vision.core.default_dirs import NEURAL_RESULTS_DIR
from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.names import filename_constructor
from mouse_vision.neural_mappers.map_param_utils import check_source_map_kw, generate_map_param_grid

def package_reliability(data_array, unit_index):
    assert len(data_array.shape) == 1
    assert data_array.shape == unit_index.shape

    # Now construct a new xarray.DataArray with the time and unit data included
    data_xarray = xr.DataArray(
        data_array, # 1 x units
        coords={"units": unit_index},
        dims=["units"]
    )

    return data_xarray

def package_temporal_reliability(data_array, time_index, unit_index):
    if len(data_array.shape) == 1:
        # we expand dimensions to add the time index
        data_array = np.expand_dims(data_array, axis=0)

    # Now construct a new xarray.DataArray with the time and unit data included
    data_xarray = xr.DataArray(
        data_array, # 1 x units
        coords={
            "time_relative_to_stimulus_onset": time_index,
            "units": unit_index
        },
        dims=[
            "time_relative_to_stimulus_onset",
            "units"
        ]
    )

    return data_xarray

def agg_interanimal_consistencies(visual_area_spec_map,
                                  source_map_kwargs=None,
                                  map_type=None,
                                  map_param_grid_idxs=None,
                                  dataset_name='neuropixels',
                                  visual_areas=VISUAL_AREAS,
                                  fit_per_target_unit=False,
                                  num_train_test_splits=10,
                                  n_ss_iter=20,
                                  n_iter=200,
                                  n_ss_imgs=None,
                                  source_unit_percentage=None,
                                  train_frac=0.5,
                                  metric_name=None,
                                  equalize_source_units=True,
                                  separate_by_animal=False,
                                  mode='pairwise',
                                  correction='spearman_brown_split_half',
                                  shorten_name=False,
                                  shorten_perc_name=False,
                                  splithalf_r_thresh=None,
                                  results_dir=NEURAL_RESULTS_DIR,
                                  train_analysis=False,
                                  save=True):

    from scipy.stats import sem

    if source_map_kwargs is None:
        save_map_kw = None # these would too long since each area would have its own parameters
        assert(map_type is not None)
        source_map_kwargs = generate_map_param_grid(map_type=map_type,
                                                    map_param_grid_idxs=map_param_grid_idxs,
                                                    visual_areas=visual_areas,
                                                    visual_area_spec_map=visual_area_spec_map,
                                                    fit_per_target_unit=fit_per_target_unit)
    else:
        save_map_kw = source_map_kwargs
        map_type = source_map_kwargs['map_type']

    # integrity checks
    source_map_kwargs = check_source_map_kw(source_map_kwargs=source_map_kwargs,
                                            visual_areas=visual_areas)

    consistencies = {}
    for v in visual_areas:
        consistencies[v] = {}
        for map_kw in source_map_kwargs[v]:
            map_kw_str = dict_to_str(map_kw)
            if separate_by_animal:
                consistencies[v][map_kw_str] = {'mean': {}, 'std': {}, 'sem': {}}
            else:
                consistencies[v][map_kw_str] = {'mean': [], 'std': [], 'sem': []}
            if mode == 'pairwise':
                spec_pairs = visual_area_spec_map[v]['specimen_pairs']
                # since the above are combinations
                spec_perms = []
                for s in spec_pairs:
                    spec_perms.append(s)
                    spec_perms.append(s[::-1])
                spec_targets = np.unique([spec_pair[1] for spec_pair in spec_perms])
            elif mode == 'holdout':
                spec_targets = [k for k in visual_area_spec_map[v].keys() if isinstance(k, np.int64)]
                assert(len(list(itertools.combinations(spec_targets, r=2))) == len(visual_area_spec_map[v]['specimen_pairs']))
                spec_perms = [('holdout', t) for t in spec_targets]
            else:
                raise ValueError
            assert(len(spec_targets) > 0)
            # loop through target neurons
            for spec_target in spec_targets:
                relevant_spec_pairs = [spec_pair for spec_pair in spec_perms if spec_pair[1] == spec_target]
                assert(len(relevant_spec_pairs) > 0)
                # aggregate across source neurons mapped to that target (as well as train/test splits and bootstraps)
                source_agg = []
                for spec_pair in relevant_spec_pairs:
                    assert(spec_pair[1] == spec_target)
                    filename = filename_constructor(source_map_kwargs=map_kw,
                                        num_source_units=visual_area_spec_map[v].get('num_source_units', None),
                                        num_train_test_splits=num_train_test_splits,
                                        n_ss_iter=n_ss_iter,
                                        n_iter=n_iter,
                                        n_ss_imgs=n_ss_imgs,
                                        source_unit_percentage=source_unit_percentage,
                                        train_frac=train_frac,
                                        metric_name=metric_name,
                                        correction=correction,
                                        shorten_name=shorten_name,
                                        shorten_perc_name=shorten_perc_name,
                                        splithalf_r_thresh=splithalf_r_thresh,
                                        names={'dataset_name': dataset_name, 'visual_area': v, 'spec_pair': spec_pair})

                    filename = os.path.join(results_dir, filename)
                    curr_rels = open_dataset(filename)
                    if isinstance(curr_rels, dict):
                        assert "train" in curr_rels.keys() and "test" in curr_rels.keys()
                        if train_analysis:
                            curr_rels = curr_rels["train"]
                        else:
                            curr_rels = curr_rels["test"]

                    if metric_name == "rsa":
                        # num_train_test_splits x bootstrap trials
                        assert(curr_rels.ndim == 2)
                        source_agg.append(curr_rels.flatten())
                    else:
                        # num_train_test_splits x bootstrap trials x num_target_neurons
                        assert(curr_rels.ndim == 3)
                        # trials x num_target_neurons (in that specimen)
                        source_agg.append(curr_rels.reshape(-1, curr_rels.shape[-1]))
                assert(len(source_agg) > 0)
                # total trials x num_target_neurons (in that visual area across all specimens)
                source_agg = np.concatenate(source_agg, axis=0)
                # mean, std, sem across all source animals and train/test splits and bootstraps for that target neuron
                target_mean = np.nanmean(source_agg, axis=0)
                target_std = np.nanstd(source_agg, axis=0)
                target_sem = sem(source_agg, axis=0, nan_policy='omit')
                if separate_by_animal:
                    consistencies[v][map_kw_str]['mean'][spec_target] = target_mean
                    consistencies[v][map_kw_str]['std'][spec_target] = target_std
                    consistencies[v][map_kw_str]['sem'][spec_target] = target_sem
                else:
                    consistencies[v][map_kw_str]['mean'].append(target_mean)
                    consistencies[v][map_kw_str]['std'].append(target_std)
                    consistencies[v][map_kw_str]['sem'].append(target_sem)

            # concatenate across all target neurons in that visual area
            if not separate_by_animal:
                for stat in consistencies[v][map_kw_str].keys():
                    if metric_name == "rsa":
                        consistencies[v][map_kw_str][stat] = np.array(consistencies[v][map_kw_str][stat])
                    else:
                        consistencies[v][map_kw_str][stat] = np.concatenate(consistencies[v][map_kw_str][stat], axis=-1)

    if save:
        save_prefix = 'interanimal_consistencies'
        if train_analysis:
            save_prefix += "_train"

        if mode != 'pairwise':
            save_prefix += '_mode{}'.format(mode)
        save_prefix += '_maptype{}_'.format(map_type) + dict_to_str({'dataset_name': dataset_name})
        save_filename = filename_constructor(num_train_test_splits=num_train_test_splits,
                                            source_map_kwargs=save_map_kw,
                                            n_ss_iter=n_ss_iter,
                                            n_iter=n_iter,
                                            n_ss_imgs=n_ss_imgs,
                                            source_unit_percentage=source_unit_percentage,
                                            train_frac=train_frac,
                                            metric_name=metric_name,
                                            correction=correction,
                                            shorten_name=shorten_name,
                                            shorten_perc_name=shorten_perc_name,
                                            splithalf_r_thresh=splithalf_r_thresh,
                                            equalize_source_units=equalize_source_units,
                                            names=save_prefix)
        save_fp = os.path.join(results_dir, save_filename)
        pickle.dump(
            consistencies,
            open(
                save_fp,
                "wb"
            )
        )
        print(f"Saved to {save_fp}")

    return consistencies
