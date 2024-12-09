import numpy as np

from mouse_vision.core.utils import iterate_dicts
from mouse_vision.core.constants import VISUAL_AREAS

def check_source_map_kw(source_map_kwargs, visual_areas=VISUAL_AREAS):
    '''
    Converts source map kwargs dictionary into a dictionary where the keys are 
    visual areas and the values are the list of kwargs one might want to try for 
    regression to units in that visual area. Convenient if you have a single 
    kwarg (first case) that you want to use uniformly for each visual area, or if
    you have a single (potentially different) kwarg per visual area and want it 
    to be turned into a singleton list for you.
    '''

    if "map_type" in source_map_kwargs.keys():
        # this is a singleton parameter, so we copy automatically to all visual areas
        source_map_kwargs = {v: [source_map_kwargs] for v in visual_areas}
    else:
        for v in visual_areas:
            assert(v in list(source_map_kwargs.keys()))
            curr_area_kw = source_map_kwargs[v]
            if not isinstance(curr_area_kw, list):
                assert(isinstance(curr_area_kw, dict) is True)
                # singleton params dict, so we wrap it as a list
                source_map_kwargs[v] = [curr_area_kw]

    return source_map_kwargs

def generate_map_param_grid(
    visual_area_spec_map=None,
    visual_areas=VISUAL_AREAS,
    fit_per_target_unit=False,
    map_type="pls",
    map_param_grid_idxs=None,
    pls_n_components_fracs=np.linspace(start=0.1, stop=1.0, num=4, endpoint=True),
    pls_n_components=None,
    pls_scale=False
):
    '''
    Given a list of  parameters (by default these are the number of components to by 
    used by PLS regression in terms of fraction of source units in each visual area), 
    returns list of kwargs to be looped over for each visual area. If you pass 
    pls_n_components directly as a list, it will use those to generate the grid. You 
    can subselect which of those parameters you want included (if there are too many 
    to be run at once on Sherlock for instance), by specifying map_param_grid_idxs.

    Output:
        ret_dict : (list) of (dict) where the dictionary has keys "map_type" and "map_kwargs"
                   and has as value another dictionary. The value for "map_type" is a neural
                   mapper string and the value for "map_kwargs" is another dictionary. This
                   dictionary contains hyperparameter name and value pairs. An example is:
                   [{"map_type": "pls", "map_kwargs": {"n_components": 10, "scale": False}}, 
                    {"map_type": "pls", "map_kwargs": {"n_components": 20, "scale": False}}]. 
    '''

    ret_dict = {}
    if map_type.lower() == "pls":
        if pls_n_components is None:
            assert(pls_n_components_fracs is not None)
            assert(visual_area_spec_map is not None)
            pls_n_components = {}
            for v in visual_areas:
                ns = visual_area_spec_map[v]["num_source_units"]
                curr_pls_n_components = [(int)(np.ceil(f*ns)) for f in pls_n_components_fracs]
                # in case any n_components frac rounds up to the same integer value
                curr_pls_n_components = list(np.unique(curr_pls_n_components))
                if map_param_grid_idxs is None:
                    pls_n_components[v] = curr_pls_n_components
                else:
                    if not isinstance(map_param_grid_idxs, list):
                        map_param_grid_idxs = [map_param_grid_idxs]
                    pls_n_components[v] = [curr_pls_n_components[map_param_grid_idx] for map_param_grid_idx in map_param_grid_idxs]
        else:
            assert pls_n_components is not None
            _pls_n_components = dict()
            for v in visual_areas:
                _pls_n_components[v] = pls_n_components
            pls_n_components = _pls_n_components

        for v in visual_areas:
            assert(v in list(pls_n_components.keys()))
            ret_dict[v] = [{"map_type": map_type.lower(), "map_kwargs": kw} for kw in iterate_dicts( \
                           {"n_components": pls_n_components[v], "scale": [pls_scale], "fit_per_target_unit": [fit_per_target_unit]})]
    elif map_type.lower() == "identity": # IdentityNeuralMap does not have hyperparameters
        for v in visual_areas:
            ret_dict[v] = [{"map_type": map_type.lower(), "map_kwargs": {}}]
    else:
        # TODO: add support for other map parameters as we get them
        raise ValueError(f"{map_type.lower()} is not supported yet.")

    return ret_dict

