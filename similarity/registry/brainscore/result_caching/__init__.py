from collections import defaultdict, OrderedDict

import inspect
import itertools
import logging
import numpy as np
import pandas as pd
import os
import pickle
import xarray as xr
from functools import wraps
from typing import Union


def get_function_identifier(function, call_args):
    module = [function.__module__, function.__name__]
    if 'self' in call_args:
        object = call_args['self']
        class_name = object.__class__.__name__
        if 'object at' in str(object):
            object = class_name
        else:
            object = f"{class_name}({str(object)})"
        module.insert(1, object)
        del call_args['self']
    module = '.'.join(module)
    strip_slashes = lambda x: str(x).replace('/', '_')
    params = ','.join(f'{key}={strip_slashes(value)}' for key, value in call_args.items())
    if params:
        function_identifier = os.path.join(module, params)
    else:
        function_identifier = module
    return function_identifier


def is_enabled(function_identifier):
    disable = os.getenv('RESULTCACHING_DISABLE', '0')
    return not _match_identifier(function_identifier, disable)


def cached_only(function_identifier):
    cachedonly = os.getenv('RESULTCACHING_CACHEDONLY', '0')
    return _match_identifier(function_identifier, cachedonly)


def _match_identifier(function_identifier, match_value):
    if match_value == '1':
        return True
    if match_value == '':
        return False
    disabled_modules = match_value.split(',')
    return any(function_identifier.startswith(disabled_module) for disabled_module in disabled_modules)


class NotCachedError(Exception):
    pass


class _Storage(object):
    def __init__(self, identifier_ignore=()):
        """
        :param [str] identifier_ignore: function parameters to ignore when building the unique function identifier.
            Different versions of the same parameter will result in the same identifier when ignored.
            Useful when the results do not depend on certain parameters.
        """
        self.identifier_ignore = identifier_ignore
        self._logger = logging.getLogger(_fullname(self))

    def __call__(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            function_identifier = self.get_function_identifier(function, call_args)
            if is_enabled(function_identifier) and self.is_stored(function_identifier):
                self._logger.debug("Loading from storage: {}".format(function_identifier))
                return self.load(function_identifier)
            if cached_only(function_identifier):
                raise NotCachedError(f"No result stored for '{function_identifier}'")
            self._logger.debug("Running function: {}".format(function_identifier))
            result = function(*args, **kwargs)
            if is_enabled(function_identifier):
                self._logger.debug("Saving to storage: {}".format(function_identifier))
                self.save(result, function_identifier)
            return result

        return wrapper

    def getcallargs(self, function, *args, **kwargs):
        call_args = inspect.getcallargs(function, *args, **kwargs)
        argspec = inspect.getfullargspec(function)
        argspec = argspec.args + \
                  ([argspec.varargs] if argspec.varargs else []) + ([argspec.varkw] if argspec.varkw else [])
        sorting = {arg: i for i, arg in enumerate(argspec)}
        return OrderedDict(sorted(call_args.items(), key=lambda pair: sorting[pair[0]]))

    def get_function_identifier(self, function, call_args):
        call_args = {key: value for key, value in call_args.items() if key not in self.identifier_ignore}
        return get_function_identifier(function, call_args)

    def is_stored(self, function_identifier):
        raise NotImplementedError()

    def load(self, function_identifier):
        raise NotImplementedError()

    def save(self, result, function_identifier):
        raise NotImplementedError()


class _DiskStorage(_Storage):
    def __init__(self, identifier_ignore=()):
        super().__init__(identifier_ignore=identifier_ignore)
        self._storage_directory = os.path.expanduser(os.getenv('RESULTCACHING_HOME', '~/.result_caching'))

    def storage_path(self, function_identifier):
        return os.path.join(self._storage_directory, function_identifier + '.pkl')

    def save(self, result, function_identifier):
        path = self.storage_path(function_identifier)
        path_dir = os.path.dirname(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        savepath_part = path + '.filepart'
        self.save_file(result, savepath_part)
        os.rename(savepath_part, path)

    def save_file(self, result, savepath_part):
        with open(savepath_part, 'wb') as f:
            pickle.dump({'data': result}, f, protocol=-1)  # highest protocol

    def is_stored(self, function_identifier):
        storage_path = self.storage_path(function_identifier)
        return os.path.isfile(storage_path)

    def load(self, function_identifier):
        path = self.storage_path(function_identifier)
        assert os.path.isfile(path)
        return self.load_file(path)

    def load_file(self, path):
        with open(path, 'rb') as f:
            return pd.read_pickle(f)['data']


class _NetcdfStorage(_DiskStorage):
    def storage_path(self, function_identifier):
        return os.path.join(self._storage_directory, function_identifier + '.nc')

    def save_file(self, result, savepath_part):
        result_coords = [coord for coord, values in self.walk_coords(result)]
        result = result.reset_index(result.indexes.keys())
        # for some reason, the above operation suffixes single-index coordinates with _
        coords = {}
        for coord, values in self.walk_coords(result):
            if coord not in result_coords:
                assert coord.endswith('_') and coord[:-1] in result_coords
                coord = coord[:-1]
            coords[coord] = values
        result = type(result)(result.values, coords=coords, dims=result.dims)
        result.to_netcdf(savepath_part)

    def load_file(self, path):
        return xr.open_dataarray(path)

    @classmethod
    def walk_coords(cls, assembly):
        """
        walks through coords and all levels, just like the `__repr__` function, yielding `(name, dims, values)`.
        """
        coords = {}

        for name, values in assembly.coords.items():
            # partly borrowed from xarray.core.formatting#summarize_coord
            is_index = name in assembly.dims
            if is_index and values.variable.level_names:
                for level in values.variable.level_names:
                    level_values = assembly.coords[level]
                    yield level, (level_values.dims, level_values.values)
            else:
                yield name, (values.dims, values.values)
        return coords


class _DictStorage(_DiskStorage):
    """
    All fields in _combine_fields are combined into one file and loaded lazily
    """

    def __init__(self, dict_key: str, *args, **kwargs):
        """
        :param dict_key: the argument representing the dictionary key.
        """
        super().__init__(*args, **kwargs)
        self._dict_key = dict_key

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            assert self._dict_key in call_args
            infile_call_args = {self._dict_key: call_args[self._dict_key]}
            function_identifier = self.get_function_identifier(function, call_args)
            stored_result, reduced_call_args = None, call_args
            if is_enabled(function_identifier) and self.is_stored(function_identifier):
                self._logger.debug(f"Loading from storage: {function_identifier}")
                stored_result = self.load(function_identifier)
                infile_missing_call_args = self.missing_call_args(infile_call_args, stored_result)
                if len(infile_missing_call_args) == 0:
                    # nothing else to run, but still need to filter
                    result = stored_result
                    reduced_call_args = None
                else:
                    # need to run more args
                    non_variable_call_args = {key: value for key, value in call_args.items() if key != self._dict_key}
                    infile_missing_call_args = {self._dict_key: infile_missing_call_args}
                    reduced_call_args = {**non_variable_call_args, **infile_missing_call_args}
                    self._logger.debug(f"Computing missing: {reduced_call_args}")
            if reduced_call_args:
                if cached_only(function_identifier):
                    raise NotCachedError(f"The following arguments for '{function_identifier}' "
                                         f"are not stored: {reduced_call_args}")
                # run function if some args are uncomputed
                self._logger.debug(f"Running function: {function_identifier}")
                result = function(**reduced_call_args)
                if not self.callargs_present(result, {self._dict_key: reduced_call_args[self._dict_key]}):
                    raise ValueError("result does not contain requested keys")
                if stored_result is not None:
                    result = self.merge_results(stored_result, result)
                # only save if new results
                if is_enabled(function_identifier):
                    self._logger.debug("Saving to storage: {}".format(function_identifier))
                    self.save(result, function_identifier)
            assert self.callargs_present(result, infile_call_args)
            result = self.filter_callargs(result, infile_call_args)
            return result

        return wrapper

    def merge_results(self, stored_result, result):
        return {**stored_result, **result}

    def callargs_present(self, result, infile_call_args):
        # make sure coords are set equal to call_args
        return len(self.missing_call_args(infile_call_args, result)) == 0

    def missing_call_args(self, call_args, data):
        assert len(call_args) == 1 and list(call_args.keys())[0] == self._dict_key
        keys = list(call_args.values())[0]
        return [key for key in keys if key not in data]

    def filter_callargs(self, data, call_args):
        assert len(call_args) == 1 and list(call_args.keys())[0] == self._dict_key
        keys = list(call_args.values())[0]
        return type(data)((key, value) for key, value in data.items() if key in keys)


class _XarrayStorage(_DiskStorage):
    """
    All fields in _combine_fields are combined into one file and loaded lazily
    """

    def __init__(self, combine_fields: Union[list, dict], sub_fields=False,
                 map_field_values=None, map_field_values_inverse=None,
                 *args, **kwargs):
        """
        :param combine_fields: fields to consider as primary keys together with the filename
            (i.e. fields not excluded by `identifier_ignore`).
        :param sub_fields: store the result right away (default, False) or only its sub-fields
        """
        super().__init__(*args, **kwargs)
        if not isinstance(combine_fields, dict):  # use identity mapping if list passed
            self._combine_fields = {field: field for field in combine_fields}
        else:
            self._combine_fields = combine_fields
        self._combine_fields_inverse = {value: key for key, value in self._combine_fields.items()}
        self._sub_fields = sub_fields
        if map_field_values:
            assert map_field_values_inverse
        self._map_field_values = map_field_values or (lambda key, value: value)
        self._map_field_values_inverse = map_field_values_inverse or (lambda key, value: value)

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = self.getcallargs(function, *args, **kwargs)
            infile_call_args = {self._combine_fields[key]: self._map_field_values(self._combine_fields[key], value)
                                for key, value in call_args.items()
                                if key in self._combine_fields}
            function_identifier = self.get_function_identifier(function, call_args)
            stored_result, reduced_call_args = None, call_args
            if is_enabled(function_identifier) and self.is_stored(function_identifier):
                self._logger.debug(f"Loading from storage: {function_identifier}")
                stored_result = self.load(function_identifier)
                missing_call_args = self.filter_coords(infile_call_args, stored_result) if not self._sub_fields \
                    else self.filter_coords(infile_call_args, getattr(stored_result, next(iter(vars(stored_result)))))
                if len(missing_call_args) == 0:
                    # nothing else to run, but still need to filter
                    result = stored_result
                    reduced_call_args = None
                else:
                    # need to run more args
                    non_variable_call_args = {key: value for key, value in call_args.items()
                                              if key not in self._combine_fields}
                    missing_call_args = {self._combine_fields_inverse[key]: self._map_field_values_inverse(key, value)
                                         for key, value in missing_call_args.items()}
                    reduced_call_args = {**non_variable_call_args, **missing_call_args}
                    self._logger.debug(f"Computing missing: {reduced_call_args}")
            if reduced_call_args:
                if cached_only(function_identifier):
                    raise NotCachedError(f"The following arguments for '{function_identifier}' "
                                         f"are not stored: {reduced_call_args}")
                self._logger.debug(f"Running function: {function_identifier}")
                # run function if some args are uncomputed
                result = function(**reduced_call_args)
                if stored_result is not None:
                    result = self.merge_results(stored_result, result)
                # only save if new results
                if is_enabled(function_identifier):
                    self._logger.debug("Saving to storage: {}".format(function_identifier))
                    self.save(result, function_identifier)
            self.ensure_callargs_present(result, infile_call_args)
            result = self.filter_callargs(result, infile_call_args)
            return result

        return wrapper

    def merge_results(self, stored_result, result):
        if not self._sub_fields:
            result = self._merge_data_arrays([stored_result, result])
        else:
            for field in vars(result):
                setattr(result, field,
                        self._merge_data_arrays([getattr(stored_result, field), getattr(result, field)]))
        return result

    def _merge_data_arrays(self, data_arrays):
        # https://stackoverflow.com/a/50125997/2225200
        merged = xr.merge([similarity.rename('z') for similarity in data_arrays])['z'].rename(None)
        # ensure same class
        return type(data_arrays[0])(merged)

    def ensure_callargs_present(self, result, infile_call_args):
        # make sure coords are set equal to call_args
        if not self._sub_fields:
            assert len(self.filter_coords(infile_call_args, result)) == 0, \
                f"{self.filter_coords(infile_call_args, result)} not present in result"
        else:
            for field in vars(result):
                assert len(self.filter_coords(infile_call_args, getattr(result, field))) == 0

    def filter_callargs(self, result, callargs):
        # filter to what function was called with
        if not self._sub_fields:
            result = self.filter_data(result, callargs)
        else:
            for field in vars(result):
                setattr(result, field, self.filter_data(getattr(result, field), callargs))
        return result

    def filter_coords(self, call_args, result):
        for key, value in call_args.items():
            assert is_iterable(value)
        combinations = [dict(zip(call_args, values)) for values in itertools.product(*call_args.values())]
        uncomputed_combinations = []
        for combination in combinations:
            combination_result = result
            combination_result = self.filter_data(combination_result, combination, single_value=True)
            if combination_result.size == 0:
                uncomputed_combinations.append(combination)
        if len(uncomputed_combinations) == 0:
            return {}
        return self._combine_call_args(uncomputed_combinations)

    def filter_data(self, data, coords, single_value=False):
        for coord, coord_value in coords.items():
            if not hasattr(data, coord):
                raise ValueError("Coord not found in data: {}".format(coord))
            # when called with a combination instantiation, coord_value will be a single item; iterable for check
            indexer = np.array([(val == coord_value) if single_value or not is_iterable(coord_value)
                                else (val in coord_value) for val in data[coord].values])
            coord_dims = data[coord].dims
            dim_indexes = {dim: slice(None) if dim not in coord_dims else np.where(indexer)[0]
                           for dim in data.dims}
            data = data.isel(**dim_indexes)
        data = data.sortby([self._build_sort_array(coord, coord_value, data)
                            for coord, coord_value in coords.items()
                            if is_iterable(coord_value) and len(coord_value) > 1])
        return data

    def _combine_call_args(self, uncomputed_combinations):
        call_args = defaultdict(list)
        for combination in uncomputed_combinations:
            for key, value in combination.items():
                call_args[key].append(value)
        return call_args

    def _build_sort_array(self, coord, coord_value, data):
        dims = data[coord].dims
        assert len(dims) == 1
        if isinstance(coord_value, tuple):
            coord_value = list(coord_value)
        s = xr.DataArray(list(range(len(coord_value))), [(coord, coord_value)])
        if dims[0] == coord:
            return s
        return s.stack(**{dims[0]: [coord]})


class _MemoryStorage(_Storage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = dict()

    def save(self, result, function_identifier):
        self.cache[function_identifier] = result

    def is_stored(self, function_identifier):
        return function_identifier in self.cache

    def load(self, function_identifier):
        return self.cache[function_identifier]


def is_iterable(x):
    try:
        iter(x)
        if isinstance(x, str):
            return False
        return True
    except TypeError:
        return False


def _fullname(obj):
    return obj.__module__ + "." + obj.__class__.__name__


def get_calling_function():
    """
    finds the calling function in many decent cases.

    Note: this function is unreliable during debugging.
    """
    # https://stackoverflow.com/a/39079070/2225200
    fr = inspect.stack()[1][0]
    co = fr.f_code
    for get in (
            lambda: fr.f_globals[co.co_name],
            lambda: getattr(fr.f_locals['self'], co.co_name),
            lambda: getattr(fr.f_locals['cls'], co.co_name),
            lambda: fr.f_back.f_locals[co.co_name],  # nested
            lambda: fr.f_back.f_locals['func'],  # decorators
            lambda: fr.f_back.f_locals['meth'],
            lambda: fr.f_back.f_locals['f'],
    ):
        try:
            func = get()
        except (KeyError, AttributeError):
            pass
        else:
            if func.__code__ == co:
                return func
    raise AttributeError("func not found")


cache = _MemoryStorage
store = _DiskStorage
store_dict = _DictStorage
store_xarray = _XarrayStorage
store_netcdf = _NetcdfStorage
