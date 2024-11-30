import itertools

from collections import OrderedDict

import numpy as np
import xarray as xr

from mouse_vision.core.utils import open_dataset
from mouse_vision.core.constants import VISUAL_AREAS, NUM_TIMEPOINTS
from mouse_vision.core.default_dirs import \
    NEUROPIX_DATA_PATH_WITH_RELS, \
    CALCIUM_DATA_PATH_WITH_RELS

def neuropix_time_avg(spec_data, splithalf_r_thresh, time_range=None):
    """
    Given a specimen's responses, average largest contiguous subset of 
    timepoints above a given split half threshold

    time_range, if given, is a list of length 2 where the first entry is the
    leftmost time point and the second entry is the rightmost time point.
    time_range must also be given in units of seconds. e.g., [0.04, 0.1] will
    result in data averaged between 40 ms and 100 ms inclusive.
    """

    def _min(x, y):
        return x if(x < y) else y

    def _max(x, y):
        return x if(x > y) else y

    # Returns the longest contiguous subarray
    def _findContig(arr, n):

        # Initialize result
        max_len = 1
        max_arr = None
        for i in range(n - 1):

            # Initialize min and max for
            # all subarrays starting with i
            mn = arr[i]
            mx = arr[i]

            # Consider all subarrays starting
            # with i and ending with j
            for j in range(i + 1, n):

                # Update min and max in
                # this subarray if needed
                mn = _min(mn, arr[j])
                mx = _max(mx, arr[j])

                # If current subarray has
                # all contiguous elements
                if ((mx - mn) == j - i):
                    curr_cont_len = mx - mn + 1
                    if curr_cont_len > max_len:
                        max_len = curr_cont_len
                        max_arr = arr[i:j+1]
                        assert(max_len == len(max_arr))

        return max_arr

    if time_range is None:
        # If time range is not given, then average across longest contiguous portion of
        # data where reliability is >= splithalf_r_thresh
        idx_arr = np.arange(0, NUM_TIMEPOINTS)
        assert(len(idx_arr) == len(spec_data.time_relative_to_stimulus_onset))
        time_arr = idx_arr[(spec_data.splithalf_r_mean.median(dim='units') >= splithalf_r_thresh).values]
        max_contig_time_arr = _findContig(arr=time_arr, n=len(time_arr))
        assert(len(max_contig_time_arr) > 0)
        spec_isel = spec_data.isel(time_relative_to_stimulus_onset=max_contig_time_arr)
        spec_time_avg = spec_isel.mean(dim='time_relative_to_stimulus_onset')
    else:
        # If time range is given, then average across the time range
        assert len(time_range) == 2
        print(f"Averaging between {time_range[0]} and {time_range[1]} seconds inclusive.")
        left, right = time_range[0], time_range[1]
        time_cond = (spec_data.time_relative_to_stimulus_onset >= left) & \
                        (spec_data.time_relative_to_stimulus_onset <= right)
        spec_sel = spec_data.where(time_cond, drop=True)
        spec_time_avg = spec_sel.mean(dim="time_relative_to_stimulus_onset")

    return spec_time_avg

def collect_neuropix_specimens(
    run_pval_ns_thresh=0.05, 
    splithalf_r_thresh=0.5, 
    min_median_splithalf_r_thresh=0.3,
    time_range=None
):
    '''
    Gather the specimens that satisfy the criteria above and average their responses across time
    '''

    ephys_data = open_dataset(NEUROPIX_DATA_PATH_WITH_RELS)
    neural_data = ephys_data['neural_data']

    collected_specimens = OrderedDict()
    for v in VISUAL_AREAS:
        collected_specimens[v] = {}
        collected_specimens[v]['specimens'] = OrderedDict()
        visual_area_data = neural_data.sel(visual_area = v)

        spec_agg = []
        for specimen_id in np.unique(visual_area_data['specimen_id'].values):
            specimen_visual_data = visual_area_data.sel(specimen_id = specimen_id)

            # We select neurons that exceed a split half reliability above threshold at some point within their timecourse
            r_timemax = specimen_visual_data.splithalf_r_mean.max(dim='time_relative_to_stimulus_onset', skipna=True)
            r_cond = ((r_timemax >= splithalf_r_thresh) & (xr.ufuncs.isfinite(r_timemax)))

            # We select neurons not significantly modulated by running speed
            speed_cond = ((specimen_visual_data.run_pval_ns > run_pval_ns_thresh) | xr.ufuncs.isnan(specimen_visual_data.run_pval_ns))

            # Now we subselect
            sel_specimen_visual_data = specimen_visual_data.where(r_cond & speed_cond, drop=True)

            # Make sure the median across units at some point exceeds min_median_splithalf_r_thresh
            if (sel_specimen_visual_data.splithalf_r_mean.median(dim='units')).max(dim='time_relative_to_stimulus_onset') >= min_median_splithalf_r_thresh:
                # Time average the largest continuous subset above threshold
                collected_specimens[v]['specimens'][specimen_id] = neuropix_time_avg(
                    spec_data=sel_specimen_visual_data,
                    splithalf_r_thresh=min_median_splithalf_r_thresh,
                    time_range=time_range
                )
                spec_agg.append(sel_specimen_visual_data.shape[-1])

        spec_agg = np.array(spec_agg)
        median_units_per_specimen = np.median(spec_agg)
        q3_units_per_specimen = np.percentile(spec_agg, 75)
        collected_specimens[v]['median_units'] = median_units_per_specimen
        collected_specimens[v]['q3_units'] = q3_units_per_specimen

    return collected_specimens

def collect_calcium_imaging_specimens(run_pval_ns_thresh=0.05, splithalf_r_thresh=0.75):
    calcium_data = open_dataset(CALCIUM_DATA_PATH_WITH_RELS)
    assert "neural_data" in calcium_data.keys()
    neural_data = calcium_data["neural_data"]

    collected_specimens = OrderedDict()
    for v in VISUAL_AREAS:
        collected_specimens[v] = {}
        collected_specimens[v]["specimens"] = OrderedDict()
        visual_area_data = neural_data.sel(visual_area=v)
        unique_specimens = np.unique(visual_area_data["specimen_id"].values)

        spec_agg = []
        for specimen_id in unique_specimens:
            specimen_visual_data = visual_area_data.sel(specimen_id=specimen_id)

            # We select neurons not significantly modulated by running
            run_condition = (specimen_visual_data["p_run_mod_ns"] > run_pval_ns_thresh) \
                                | xr.ufuncs.isnan(specimen_visual_data["p_run_mod_ns"])

            # We select neurons that exceed a split half reliability above threshold
            reliability_condition = (specimen_visual_data["reliability_mean"] >= splithalf_r_thresh)

            # Now, subselect the data
            sel_specimen_visual_data = specimen_visual_data.where(run_condition & reliability_condition, drop=True)
            collected_specimens[v]["specimens"][specimen_id] = sel_specimen_visual_data

            # Append number of units
            num_cells = len(np.unique(sel_specimen_visual_data["cell_specimen_id"].values))
            assert num_cells == sel_specimen_visual_data.shape[-1]
            spec_agg.append(sel_specimen_visual_data.shape[-1])

        spec_agg = np.array(spec_agg)
        median_units_per_specimen = np.median(spec_agg)
        q3_units_per_specimen = np.percentile(spec_agg, 75)
        collected_specimens[v]["median_units"] = median_units_per_specimen
        collected_specimens[v]["q3_units"] = q3_units_per_specimen

    return collected_specimens

def _gen_spec_map(collected_specimens, equalize_source_units=True, area_unit_thresh_val=None):
    """
    Inputs:
        collected_specimens   : (dict) with one key for each visual area. The value is another 
                                dictionary with keys: ["median_units", "q3_units", "specimens"]. 
                                The value for the "specimens" key is the specimen ID and the value 
                                for that is the xarray for neural responses.
        equalize_source_units : (boolean) take minimum number of source units across animals for 
                                interanimal comparison.
        area_unit_thresh_val  : 
    """
    for v in VISUAL_AREAS:
        assert v in collected_specimens.keys()
        assert "median_units" in collected_specimens[v].keys()
        assert "q3_units" in collected_specimens[v].keys()
        assert "specimens" in collected_specimens[v].keys()

    # We now select specimens with at least a certain number of units to make the comparison nontrivial
    spec_map = OrderedDict()
    for v in VISUAL_AREAS:
        spec_map[v] = OrderedDict()
        if area_unit_thresh_val is None:
            area_unit_thresh = (int)(np.ceil(collected_specimens[v]["q3_units"]))
        elif isinstance(area_unit_thresh_val, str): # e.g. 'median_units'
            area_unit_thresh = (int)(np.ceil(collected_specimens[v][area_unit_thresh_val]))
        else:
            area_unit_thresh = area_unit_thresh_val

        min_units = np.inf
        for s in collected_specimens[v]["specimens"].keys():
            if len(collected_specimens[v]["specimens"][s].units) >= area_unit_thresh:
                spec_map[v][s] = collected_specimens[v]["specimens"][s]
                curr_num_units = len(spec_map[v][s].units)
                if min_units > curr_num_units:
                    min_units = curr_num_units

        spec_map[v]["specimen_pairs"] = list(itertools.combinations(list(spec_map[v].keys()), r=2))

        # This will ensure that we use the same number of source units per specimen within a given visual area
        if equalize_source_units:
            spec_map[v]["num_source_units"] = min_units

    return spec_map

def generate_spec_map(
    dataset_name="neuropixels",
    equalize_source_units=True,
    area_unit_thresh_val=None,
    time_range=None,
    **kwargs
):
    """
    Generates the specimen map for each visual area depending on the dataset

    time_range : either None or a list of length 2. It is passed to
                 collect_neuropix_specimens in order to determine whether or
                 not to average over a pre-specified time interval.
    """

    if dataset_name == "neuropixels":
        collected_specimens = collect_neuropix_specimens(time_range=time_range, **kwargs)
    elif dataset_name == "calcium":
        collected_specimens = collect_calcium_imaging_specimens(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} does not exist.")

    spec_map = _gen_spec_map(
        collected_specimens, 
        equalize_source_units=equalize_source_units, 
        area_unit_thresh_val=area_unit_thresh_val
    )

    return spec_map


