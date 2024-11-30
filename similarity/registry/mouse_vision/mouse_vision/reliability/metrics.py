import numpy as np
import xarray as xr
import scipy.stats as stats
from joblib import Parallel, delayed
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances, PAIRWISE_DISTANCE_FUNCTIONS

DEF_BITER = 900    # default bootstrapping iters
DEF_SSITER = 100    # default subsample iterations

def upper_tri(X):
    '''Returns upper triangular part due to symmetry of RSA.
        Excludes diagonal as generally recommended: https://www.sciencedirect.com/science/article/pii/S1053811916308059'''
    return X[np.triu_indices_from(X, k=1)]

def rdm(X):
    return 1 - np.corrcoef(X)

def rsa(X, Y):
    assert X.ndim == 2
    if isinstance(X, xr.DataArray):
        # provide an extra layer of security for xarrays
        assert X.dims[0] == 'frame_id'
        assert X.dims[1] == 'units'
    assert Y.ndim == 2
    if isinstance(Y, xr.DataArray):
        # provide an extra layer of security for xarrays
        assert Y.dims[0] == 'frame_id'
        assert Y.dims[1] == 'units'
    rdm_X = upper_tri(rdm(X))
    rdm_X = rdm_X.flatten()
    rdm_Y = upper_tri(rdm(Y))
    rdm_Y = rdm_Y.flatten()
    return stats.pearsonr(rdm_X, rdm_Y)[0]

# helper functions for sanity checks
def _check_list(X):
    if (X is not None) and (not isinstance(X, list)):
        return [X]
    else:
        return X

def _check_ctr_data(N, center, N_ctr=None):
    if N is not None:
        n_img = 0
        n_neu = N[0].shape[-1]

        for N0 in N:
            assert (N0.ndim == 2) or (N0.ndim == 3)
            if isinstance(N0, xr.DataArray) and (N0.ndim == 3):
                # provide an extra layer of security for xarrays
                assert N0.dims[0] == 'trials'
                assert N0.dims[1] == 'frame_id'
                assert N0.dims[2] == 'units'
            assert N0.shape[-1] == n_neu
            if N0.ndim == 3:
                n_img += N0.shape[1]
            elif N0.ndim == 2:
                n_img += N0.shape[0]

        if N_ctr is None:
            if N0.ndim == 3:
                N_ctr = [center(N0, axis=0) for N0 in N]
            elif N0.ndim == 2:
                # deterministic model features so there is no trials dimension
                # to center over
                N_ctr = [N0 for N0 in N]

        assert(len(N) == len(N_ctr))
        assert np.concatenate(N_ctr, axis=0).shape == (n_img, n_neu)
    return N_ctr

def _check_source_N(target_N, target_Nctr,
                    source_N=None, source_Nctr=None):
    if source_N is not None:
        assert(len(source_N) == len(target_N))
        assert(source_Nctr is not None)
        target_n_img = np.concatenate(target_Nctr, axis=0).shape[0]
        source_n_img = np.concatenate(source_Nctr, axis=0).shape[0]
        assert(source_n_img == target_n_img)

def _check_source_subsample(source_N=None, num_source_units=None):
    if (source_N is not None) and (num_source_units is not None):
        assert(num_source_units <= source_N[0].shape[-1])

def _check_train_test_idx(train_img_idx=None,
                          test_img_idx=None,
                          source_map=None,
                          num_image_types=1):
    if (train_img_idx is not None) or (test_img_idx is not None):
        assert((train_img_idx is not None) and (test_img_idx is not None))
        assert(source_map is not None)
        train_img_idx = _check_list(train_img_idx)
        test_img_idx = _check_list(test_img_idx)
        # we specify train/test indices per image type
        assert(len(train_img_idx) == num_image_types)
        assert(len(train_img_idx) == len(test_img_idx))
        # check that train and test indices are are non-overlapping
        for curr_train_img_idx, curr_test_img_idx in zip(train_img_idx, test_img_idx):
            assert(set(curr_train_img_idx).isdisjoint(curr_test_img_idx) is True)
    return train_img_idx, test_img_idx

# final TODO: after testing, add documentation to all functions include helpers

# takes in: source_N (default None) and target_N (required and will be what is now N),
# source_mapping_function (default identity)
# train_img_idx (None: all images by default) and test_img_idx (None: all images by default, else required to be non-overlapping with train) for a single train test split
# number of source neurons: (None: all neurons by default but can subsample if specified to be less than all of them)
# default behavior will therefore be split half reliability computed per target neuron
def noise_estimation(target_N,
                     parallelize_per_target_unit=True,
                     source_N=None,
                     source_map_kwargs=None,
                     train_img_idx=None, test_img_idx=None,
                     num_source_units=None,
                     metric='pearsonr', mode='spearman_brown_split_half',
                     target_Nctr=None, source_Nctr=None,
                     center=np.nanmean,
                     summary_center=np.mean, summary_spread=np.std,
                     report_train=False,
                     sync=True, n_jobs=1, n_iter=DEF_BITER, n_ss_iter=DEF_SSITER, start_seed=0):

    """Estimate the self-self consistencies (e.g., correlation coeffs) of
    individual neurons across trials for the given images --- i.e., internal
    consistency.
    Parameters
    ---------
    N: list of array-like
        List of neuronal data.  Each item should be an array and should
        have the shape of (# of trials, # of images, # of neurons).
        Each item represents a subset of neuronal data for different
        images (e.g., V0, V3, V6).  While the # of trials can be different,
        the neurons must be the same across all items in ``N``.
    metric: string, or callable, default='pearsonr'
        Which metric to use to compute the "internal consistency."
        Supported:
            * 'pearsonr' and 'spearmanr'
            * All valid `metric` in sklearn's pairwise_distances()
            * callable, which takes two vectors and gives a distance between
              the two.
    mode: string, default='spearman_brown_split_half'
        Which method to use to compute the "internal consistency."
        Supported:
            * 'bootstrap': This reconstructs two sets of neuronal data by
                bootstrapping over trials and compares those two replicas.
            * 'bootstrap_assume_true_ctr': reconstructs neuronal data
                by bootstrapping over trials and compares with the fixed,
                estimated true central tendency (``Nctr``, see below).
                This assumes we know the true central tendency (which is
                usually NOT TRUE) and therefore always gives the highest
                internal consistency values among all supported modes here.
            * 'spearman_brown_split_half': splits the data into halves,
                computes the consistency between **non-overlapping** halves,
                and applies Spearman-Brown prediction formula.  This
                typically gives the low bound of the internal consistency.
            * 'spearman_brown_subsample_half': reconstructs two sets of
                data by two independent subsamplings of the original data
                into half (without replacement), computes the consistency
                between two sets (which can potentially share some
                **overlapping trials**), and applies Spearman-Brown formula.
                Empirically, this gives similar values as
                'bootstrap_assume_true_ctr' does, BUT DOES NOT HAVE
                CLEAR MATHEMATICAL FOUNDATION.  USE WITH CARE.
            * 'spearman_brown_subsample_half_replacement': Same as
                'spearman_brown_subsample_half' but this subsamples trials
                with **replacement** as in typical bootstrapping.
                This is essentially making bootstrap replicas of
                the original data and running 'spearman_brown_split_half'
                over those replicas.  As expected, this gives very
                similar internal consistency values as 'bootstrap' does.
    Nctr: list of array-like, shape=(# of images, # of neurons), default=None
        If given, ``Nctr`` will be used as the true central tendency values
        of neuronal responses to images (across trials).  Otherwise,
        the mean across trials will be computed and used by default.
    sync: bool, default=True
        If True, approximate time synchrony across images will be maintained.
        (aka non-shuffled)
    center: callable, default=np.mean
        The function used to estimate the central tendency of responses
        to images for a given neuron across (reconstructed) trials.  This must
        accept the keyword (named) argument 'axis' like np.mean().
    summary_center: callable, or 'raw', default=np.mean
        The function used to estimate the central tendency across different
        reconstructions of the data (e.g., bootstrapping samples).  If 'raw' is
        given, the raw values will be returned.
    summary_spread: callable, default=np.std
        The function used to estimate the spread across different
        reconstructions of the data.
    n_iter: int
        The # of reconstructions of the original data (e.g., bootstrap
        samplings, different split-halves).
    Returns
    -------
    If 'summary_center' is not 'raw':
        r: array-like, shape=(# of neurons)
            Contains each neuron's estimated self-self consistency across
            the given images.
        s: array-like, shape=(# of neurons)
            Spread of self-self consistencies of neurons.
    Else:
        rs: array-like, shape=(# of neurons, ``n_iter``)
            Contains each neuron's estimated self-self consistency across
            the given images over different reconstructions of the data.
    """

    assert(start_seed >= 0)
    target_N = _check_list(target_N)
    assert target_N[0].ndim == 3
    source_N = _check_list(source_N)
    target_Nctr = _check_ctr_data(target_N, N_ctr=target_Nctr, center=center)
    target_n_neu = target_N[0].shape[-1]
    source_Nctr = _check_ctr_data(source_N, N_ctr=source_Nctr, center=center)
    _check_source_N(target_N=target_N, target_Nctr=target_Nctr,
                    source_N=source_N, source_Nctr=source_Nctr)
    _check_source_subsample(source_N=source_N, num_source_units=num_source_units)
    train_img_idx, test_img_idx = _check_train_test_idx(train_img_idx=train_img_idx,
                                                        test_img_idx=test_img_idx,
                                                        source_map=source_map_kwargs,
                                                        num_image_types=len(target_N))

    # check mode
    assert mode in ['bootstrap', 'bootstrap_assume_true_ctr',
            'spearman_brown_split_half',
            'spearman_brown_split_half_denominator',
            'spearman_brown_subsample_half',
            'spearman_brown_subsample_half_replacement']
    # check metric
    if metric in ['spearman', 'spearmanr', 'pearson', 'pearsonr', 'rsa']:
        pass
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        pass
    elif callable(metric):
        pass
    elif hasattr(distance, metric):
        pass
    else:
        raise ValueError('Unknown "metric"')

    if 'spearman_brown' in mode and metric not in ['pearson', 'pearsonr']:
        from warnings import warn
        warn('Using Spearman-Brown prediction formula for metrics other than' \
                "Pearson's correlation coefficient is NOT intended, " \
                'and therefore the returned internal consistencies' \
                'WILL BE MATHEMATICALLY UNGROUNDED.'
                )

    if mode in ['bootstrap', 'spearman_brown_subsample_half',
            'spearman_brown_subsample_half_replacement']:
        # only this amount is needed because
        # "spearman_brown_subsample_half" and
        # "spearman_brown_subsample_half_replacement"
        # are computed in pair-wise fashion
        n_iter = int(np.sqrt(n_iter))
    assert n_iter > 1

    if (source_N is None) or ((source_N is not None) and ((num_source_units is None) or ((num_source_units is not None) and (num_source_units == source_N[0].shape[-1])))):
        n_ss_iter = 1 # no source subsampling so no need to run for multiple subsample iterations
        num_source_units = None # in the event it is specified but is equal to the actual number of source neurons so they aren't shuffled

    # number crunching!!!
    if parallelize_per_target_unit:
        if type(metric) is str:
            assert(metric != 'rsa') # RSA is correlated across ALL neurons per image pairs, so cannot be done in parallel
        print('Parallelizing across {} target neurons, using {} jobs'.format(target_n_neu, n_jobs))
        results = Parallel(n_jobs=n_jobs)(delayed(_noise_estimation_worker)
                (target_N=[np.expand_dims(target_N0[:, :, ni], axis=-1) for target_N0 in target_N],
                 target_Nctr=[np.expand_dims(target_Nctr0[:, ni], axis=-1) for target_Nctr0 in target_Nctr],
                 metric=metric,
                 mode=mode,
                 center=center,
                 source_N=source_N,
                 source_Nctr=source_Nctr,
                 source_map_kwargs=source_map_kwargs,
                 train_img_idx=train_img_idx, test_img_idx=test_img_idx,
                 num_source_units=num_source_units,
                 report_train=report_train,
                 sync=sync,
                 seed=start_seed,
                 n_ss_iter=n_ss_iter,
                 n_iter=n_iter) for ni in range(target_n_neu))

        # aggregate across target neurons prior to summarizing
        if report_train:
            results_test, results_train = results
            results_test = np.concatenate(results_test, axis=-1)
            results_train = np.concatenate(results_train, axis=-1)
        else:
            results = np.concatenate(results, axis=-1)
    else:
        # if you instead want to parallelize across bootstrap iterations (especially useful when you have a source mapping)
        if n_ss_iter > n_iter:
            # parallelize across subsampled source units if subsampling and if this exceeds the number of bootstrapped trials
            num_trials = n_ss_iter
            n_ss_iter = 1
            print('Parallelizing across {} trials of subsampled source units, using {} jobs'.format(num_trials, n_jobs))
        else:
            # otherwise parallelize across bootstrapped trials if either not subsampling the source units or exceeds the number of subsampling trials
            num_trials = n_iter
            n_iter = 1
            print('Parallelizing across {} bootstrapped trials, using {} jobs'.format(num_trials, n_jobs))

        results = Parallel(n_jobs=n_jobs)(delayed(_noise_estimation_worker)
                (target_N=target_N,
                 target_Nctr=target_Nctr,
                 metric=metric,
                 mode=mode,
                 center=center,
                 source_N=source_N,
                 source_Nctr=source_Nctr,
                 source_map_kwargs=source_map_kwargs,
                 train_img_idx=train_img_idx, test_img_idx=test_img_idx,
                 num_source_units=num_source_units,
                 report_train=report_train,
                 sync=sync,
                 seed=si, # we change the random seed per trial to have the same effect as when we did them both together
                 n_ss_iter=n_ss_iter,
                 n_iter=n_iter) for si in range(start_seed, num_trials+start_seed))

        # aggregate across trials prior to summarizing
        if report_train:
            assert len(results) == num_trials
            results_test = []
            results_train = []
            for _res in results:
                assert len(_res) == 2
                results_test.append(_res[0]) # first result is the test
                results_train.append(_res[1]) # second result is the train
            results_test = np.concatenate(results_test, axis=0)
            results_train = np.concatenate(results_train, axis=0)
        else:
            results = np.concatenate(results, axis=0)

    # if distributions are requested... (undocumented!!)
    if summary_center == 'raw':
        if report_train:
            return results_test, results_train
        else:
            return results

    if report_train:
        return summary_center(results_test, axis=0), summary_spread(results_test, axis=0), summary_center(results_train, axis=0), summary_spread(results_train, axis=0)
    else:
        return summary_center(results, axis=0), summary_spread(results, axis=0)

def _spearmanr_helper(x, y):
    return stats.spearmanr(x, y)[0]

def _sample_helper(N0,
                   nctr0,
                   sample_func,
                   mode,
                   img_type_idx=None,
                   sample_ni=None):
    x = None
    y = None
    if N0 is not None:
        if img_type_idx is not None:
            N0 = N0[img_type_idx]
        if N0.ndim == 2:
            # model features so no need to sample by trials
            x = N0
            y = N0
        else:
            assert N0.ndim == 3
            x, y = sample_func(N0)
        if mode == 'bootstrap_assume_true_ctr':
            assert(nctr0 is not None)
            if img_type_idx is not None:
                nctr0 = nctr0[img_type_idx]
            y = nctr0
        if sample_ni is not None:
            assert(len(x.shape) == 2)
            assert(len(y.shape) == 2)
            x = x[:, sample_ni]
            y = y[:, sample_ni]
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=-1)
            if len(y.shape) == 1:
                y = np.expand_dims(y, axis=-1)
    return x, y

def _train_test_sp(inp, train_idx=None, test_idx=None, img_type_idx=None):
    train = inp
    test = inp
    if inp is None:
        # to appease extend, which is called on these
        train = [train]
        test = [test]
    elif (train_idx is not None) and (test_idx is not None):
        if img_type_idx is not None:
            train_idx = train_idx[img_type_idx]
            test_idx = test_idx[img_type_idx]
        train = inp[train_idx]
        test = inp[test_idx]
    return train, test

def _map_regress(X_train,
                 y_train,
                 X_test,
                 report_train=False,
                 map_kwargs=None):
    pred_train = X_train
    pred_test = X_test
    if map_kwargs is not None:
        from mouse_vision.neural_mappers import PipelineNeuralMap
        map_func = PipelineNeuralMap(**map_kwargs)
        map_func.fit(X_train, y_train)
        pred_test = map_func.predict(X_test)
        if report_train:
            pred_train = map_func.predict(X_train)
    if report_train:
        return pred_test, pred_train
    else:
        return pred_test

def _compute_metric_across_iters(Xsmp,
                                 Ysmp,
                                 metric,
                                 mode,
                                 corrector,
                                 metric_name=None):

    # -- get the numbers and done
    if (mode != 'spearman_brown_split_half') and (mode != 'spearman_brown_split_half_denominator'):
        ds = np.ravel(pairwise_distances(Ysmp, Xsmp, metric=metric))
    else:
        if metric_name != 'rsa':
            # for RSA this constraint does not need to be enforced
            # (e.g. under an identity mapping)
            assert Xsmp.shape == Ysmp.shape
        # Essentially, this equals to taking the diagonalof the above.
        ds = np.array([metric(x, y) for x, y in zip(Xsmp, Ysmp)])

    ds = corrector(ds)

    return ds

def _return_curr(X, curr_idx, agg_per_neuron=True, agg_axis=-1):
    assert X.ndim == 3
    if agg_per_neuron:
        return X[:, :, curr_idx] # in (n_reps, images)
    else:
        # we append the 1 since _compute_metric_across_iters loops across the first dimension internally
        # and we don't want it to loop through images but rather feed an images x target_n_neu matrix to the metric in this case
        return np.expand_dims(X[curr_idx, :, :], axis=agg_axis) # in (1, images, target_n_neu)

def _noise_estimation_worker(target_N, target_Nctr,
                             metric,
                             mode,
                             center,
                             source_N=None, source_Nctr=None,
                             source_map_kwargs=None,
                             train_img_idx=None, test_img_idx=None,
                             num_source_units=None,
                             seed=0,
                             n_iter=DEF_BITER,
                             n_ss_iter=DEF_SSITER,
                             report_train=False,
                             sync=True):
    """Helper function for noise_estimation().
    N: list of one neuron's responses. Each element's shape =
        (# of reps, # of images)
    nctr: the central tendencies of the neuron's responses
    """
    rng = np.random.RandomState(seed)
    n_img = np.concatenate(target_Nctr, axis=0).shape[0]
    if test_img_idx is not None:
        n_img = np.concatenate(test_img_idx, axis=0).shape[0]
    target_n_neu = target_Nctr[0].shape[-1]
    # -- deal with metric and mode
    # various "correctors"
    def corrector_dummy(X):
        return X

    def corrector_pearsonr(X):
        X = 1. - X
        return X

    agg_per_neuron = True # compute the metric separately per neuron, across images
    corrector = corrector_dummy   # do nothing by default

    metric_name = None
    if metric in ['spearman', 'spearmanr']:
        metric_name = metric
        metric = _spearmanr_helper   # but this will be slow.
    elif metric in ['pearson', 'pearsonr']:
        # NOTE: Pearson r will be computed by sklearn's
        # pairwise_distances() with metric="correlation" for
        # efficiency.  Because "correlation" is 1 - pearsonr,
        # the returned values **MUST** be corrected as below.
        metric_name = metric
        metric = 'correlation'
        corrector = corrector_pearsonr
    elif metric == 'rsa':
        metric_name = 'rsa'
        metric = rsa
        agg_per_neuron = False # RSA is correlated across ALL neurons per image pairs, so cannot be computed separately per neuron

    if (mode == 'spearman_brown_split_half' or mode == 'spearman_brown_split_half_denominator') and (type(metric) is str):
        if metric in PAIRWISE_DISTANCE_FUNCTIONS:
            metric = PAIRWISE_DISTANCE_FUNCTIONS[metric]
        elif not callable(metric):
            metric = getattr(distance, metric)

    # -- various subsampling helper functions
    def bsample_sync(M, n_div=1):
        n_rep = M.shape[0]
        ri = rng.randint(n_rep, size=(n_rep // n_div))
        return center(M[ri], axis=0), []   # should be 1D vector

    def bsample_async(M, n_div=1):
        n_rep = M.shape[0]
        M_T = np.swapaxes(M, axis1=1, axis2=0) # in (# img, # reps)
        x = [e[rng.randint(n_rep, size=(n_rep // n_div))] for e in M_T]
        x = np.array(x)
        x = np.swapaxes(x, axis1=0, axis2=1) # in (# reps, # imgs)
        return center(x, axis=0), []

    def bsamplehalf_sync(M):
        return bsample_sync(M, n_div=2)[0], bsample_sync(M, n_div=2)[0]

    def bsamplehalf_async(M):
        return bsample_async(M, n_div=2)[0], bsample_async(M, n_div=2)[0]

    def bsamplefull_sync(M):
        return bsample_sync(M)[0], bsample_sync(M)[0]

    def bsamplefull_async(M):
        return bsample_async(M)[0], bsample_async(M)[0]

    def splithalf_sync(M):
        n_rep = M.shape[0]
        ri = list(range(n_rep))
        rng.shuffle(ri)   # without replacement
        sphf_n_rep = n_rep // 2
        ri1 = ri[:sphf_n_rep]
        ri2 = ri[sphf_n_rep:]
        return center(M[ri1], axis=0), center(M[ri2], axis=0)

    def splithalf_async(M):
        n_rep = M.shape[0]
        x = []
        y = []
        ri = list(range(n_rep))
        M_T = np.swapaxes(M, axis1=1, axis2=0) # in (# img, # reps)
        for e in M_T:   # image major
            rng.shuffle(ri)  # without replacement
            sphf_n_rep = n_rep // 2
            ri1 = ri[:sphf_n_rep]
            ri2 = ri[sphf_n_rep:]
            x.append(e[ri1])
            y.append(e[ri2])
        x = np.array(x)
        x = np.swapaxes(x, axis1=0, axis2=1) # in (# reps, # imgs)
        y = np.array(y)
        y = np.swapaxes(y, axis1=0, axis2=1) # in (# reps, # imgs)
        return center(x, axis=0), center(y, axis=0)

    SAMPLE_FUNC_REGISTRY = {
            ('bootstrap', True): bsamplefull_sync,
            ('bootstrap', False): bsamplefull_async,
            ('bootstrap_assume_true_ctr', True): bsample_sync,
            ('bootstrap_assume_true_ctr', False): bsample_async,
            ('spearman_brown_split_half', True): splithalf_sync,
            ('spearman_brown_split_half', False): splithalf_async,
            ('spearman_brown_split_half_denominator', True): splithalf_sync,
            ('spearman_brown_split_half_denominator', False): splithalf_async,
            ('spearman_brown_subsample_half', True): splithalf_sync,
            ('spearman_brown_subsample_half', False): splithalf_async,
            ('spearman_brown_subsample_half_replacement', True): bsamplehalf_sync,
            ('spearman_brown_subsample_half_replacement', False): bsamplehalf_async,
        }
    sample = SAMPLE_FUNC_REGISTRY[mode, sync]

    # -- reconstruct the data many times
    if report_train:
        target_Xsmp_train = []
        target_Ysmp_train = []
        source_target_Xsmp_train = []
        source_target_Ysmp_train = []
        source_mapping_Xsmp_train = []
        source_mapping_Ysmp_train = []

    target_Xsmp = []
    target_Ysmp = []
    source_target_Xsmp = []
    source_target_Ysmp = []
    source_mapping_Xsmp = []
    source_mapping_Ysmp = []

    for _ in range(n_ss_iter):
        source_sample_ni = None
        if (source_N is not None) and (num_source_units is not None):
            # fixed subsample source neuron indices per subsample iteration
            source_n_neu = source_N[0].shape[-1]
            source_ni = list(range(source_n_neu))
            rng.shuffle(source_ni)  # without replacement
            source_sample_ni = source_ni[:num_source_units]

        for _ in range(n_iter):
            target_xs_train = []
            target_xs_test = []
            target_ys_train = []
            target_ys_test = []
            source_xs_train = []
            source_xs_test = []
            source_ys_train = []
            source_ys_test = []
            for img_type_idx, _ in enumerate(target_N):
                # sample from trials and subselect neurons if need be
                target_x, target_y = _sample_helper(N0=target_N, nctr0=target_Nctr,
                                                    sample_func=sample, mode=mode,
                                                    img_type_idx=img_type_idx)

                source_x, source_y = _sample_helper(N0=source_N, nctr0=source_Nctr,
                                                    sample_func=sample, mode=mode,
                                                    img_type_idx=img_type_idx,
                                                    sample_ni=source_sample_ni)

                # get train/test splits if defined
                target_x_train, target_x_test = _train_test_sp(inp=target_x,
                                                               train_idx=train_img_idx,
                                                               test_idx=test_img_idx,
                                                               img_type_idx=img_type_idx)
                target_y_train, target_y_test = _train_test_sp(inp=target_y,
                                                               train_idx=train_img_idx,
                                                               test_idx=test_img_idx,
                                                               img_type_idx=img_type_idx)
                source_x_train, source_x_test = _train_test_sp(inp=source_x,
                                                               train_idx=train_img_idx,
                                                               test_idx=test_img_idx,
                                                               img_type_idx=img_type_idx)
                source_y_train, source_y_test = _train_test_sp(inp=source_y,
                                                               train_idx=train_img_idx,
                                                               test_idx=test_img_idx,
                                                               img_type_idx=img_type_idx)

                # aggregate across images
                target_xs_train.append(target_x_train)
                target_xs_test.append(target_x_test)
                target_ys_train.append(target_y_train)
                target_ys_test.append(target_y_test)
                source_xs_train.append(source_x_train)
                source_xs_test.append(source_x_test)
                source_ys_train.append(source_y_train)
                source_ys_test.append(source_y_test)

            # convert to numpy arrays
            target_xs_train = np.concatenate(target_xs_train, axis=0)
            target_xs_test = np.concatenate(target_xs_test, axis=0)
            target_ys_train = np.concatenate(target_ys_train, axis=0)
            target_ys_test = np.concatenate(target_ys_test, axis=0)
            source_xs_train = np.concatenate(source_xs_train, axis=0)
            source_xs_test = np.concatenate(source_xs_test, axis=0)
            source_ys_train = np.concatenate(source_ys_train, axis=0)
            source_ys_test = np.concatenate(source_ys_test, axis=0)

            if report_train:
                source_target_xs, source_target_xs_train = _map_regress(
                    X_train=source_xs_train,
                    y_train=target_xs_train,
                    X_test=source_xs_test,
                    report_train=report_train,
                    map_kwargs=source_map_kwargs
                )

                source_target_ys, source_target_ys_train = _map_regress(
                    map_kwargs=source_map_kwargs,
                    X_train=source_ys_train,
                    y_train=target_ys_train,
                    X_test=source_ys_test,
                    report_train=report_train
                )
            else:
                source_target_xs = _map_regress(map_kwargs=source_map_kwargs,
                                                X_train=source_xs_train,
                                                y_train=target_xs_train,
                                                X_test=source_xs_test)

                source_target_ys = _map_regress(map_kwargs=source_map_kwargs,
                                                X_train=source_ys_train,
                                                y_train=target_ys_train,
                                                X_test=source_ys_test)

            assert target_xs_test.shape == (n_img, target_n_neu)
            if report_train:
                assert target_xs_train.shape == (n_img, target_n_neu)
                target_Xsmp_train.append(target_xs_train)
                target_Ysmp_train.append(target_ys_train)

            target_Xsmp.append(target_xs_test)
            target_Ysmp.append(target_ys_test)

            if (source_N is not None) and (metric_name != 'rsa'):
                # for RSA this constraint does not need to be enforced
                # (e.g. under an identity mapping source_target_xs == (n_img, source_n_neu))
                assert source_target_xs.shape == (n_img, target_n_neu)
                if report_train:
                    assert source_target_xs_train.shape == (n_img, target_n_neu)

            if report_train:
                source_target_Xsmp_train.append(source_target_xs_train)
                source_target_Ysmp_train.append(target_ys_train)
                source_mapping_Xsmp_train.append(source_target_xs_train)
                source_mapping_Ysmp_train.append(source_target_ys_train)

            source_target_Xsmp.append(source_target_xs)
            source_target_Ysmp.append(target_ys_test)
            source_mapping_Xsmp.append(source_target_xs)
            source_mapping_Ysmp.append(source_target_ys)

    if report_train:
        target_Xsmp_train = np.array(target_Xsmp_train) # in (n_reps, images, target_n_neu)
        target_Ysmp_train = np.array(target_Ysmp_train) # in (n_reps, images, target_n_neu)

    target_Xsmp = np.array(target_Xsmp) # in (n_reps, images, target_n_neu)
    target_Ysmp = np.array(target_Ysmp) # in (n_reps, images, target_n_neu)

    if report_train:
        ds_agg_train = []

    ds_agg = []
    if agg_per_neuron:
        agg_axis = -1
        assert target_n_neu == target_Xsmp.shape[agg_axis]
        if report_train:
            assert target_n_neu == target_Xsmp_train.shape[agg_axis]
    else:
        agg_axis = 0
    num_aggs = target_Xsmp.shape[agg_axis]
    if report_train:
        assert num_aggs == target_Xsmp_train.shape[agg_axis]
    for ni in range(num_aggs):
        # subselect
        if report_train:
            curr_target_Xsmp_train = _return_curr(target_Xsmp_train,
                                                curr_idx=ni,
                                                agg_per_neuron=agg_per_neuron,
                                                agg_axis=agg_axis)
            curr_target_Ysmp_train = _return_curr(target_Ysmp_train,
                                                curr_idx=ni,
                                                agg_per_neuron=agg_per_neuron,
                                                agg_axis=agg_axis)

            ds_train = _compute_metric_across_iters(Xsmp=curr_target_Xsmp_train,
                                                  Ysmp=curr_target_Ysmp_train,
                                                  metric=metric,
                                                  metric_name=metric_name,
                                                  mode=mode,
                                                  corrector=corrector)

        curr_target_Xsmp = _return_curr(target_Xsmp,
                                        curr_idx=ni,
                                        agg_per_neuron=agg_per_neuron,
                                        agg_axis=agg_axis)
        curr_target_Ysmp = _return_curr(target_Ysmp,
                                        curr_idx=ni,
                                        agg_per_neuron=agg_per_neuron,
                                        agg_axis=agg_axis)

        ds = _compute_metric_across_iters(Xsmp=curr_target_Xsmp,
                                          Ysmp=curr_target_Ysmp,
                                          metric=metric,
                                          metric_name=metric_name,
                                          mode=mode,
                                          corrector=corrector)

        if source_N is not None:
            # we do the interanimal function with the given metric and mode
            if report_train:
                source_target_Xsmp_train = np.array(source_target_Xsmp_train) # in (n_reps, images, target_n_neu)
                source_target_Ysmp_train = np.array(source_target_Ysmp_train) # in (n_reps, images, target_n_neu)
                source_mapping_Xsmp_train = np.array(source_mapping_Xsmp_train) # in (n_reps, images, target_n_neu)
                source_mapping_Ysmp_train = np.array(source_mapping_Ysmp_train) # in (n_reps, images, target_n_neu)
                # subselect
                curr_source_target_Xsmp_train = _return_curr(source_target_Xsmp_train,
                                                       curr_idx=ni,
                                                       agg_per_neuron=agg_per_neuron,
                                                       agg_axis=agg_axis)
                curr_source_target_Ysmp_train = _return_curr(source_target_Ysmp_train,
                                                       curr_idx=ni,
                                                       agg_per_neuron=agg_per_neuron,
                                                       agg_axis=agg_axis)
                curr_source_mapping_Xsmp_train = _return_curr(source_mapping_Xsmp_train,
                                                        curr_idx=ni,
                                                        agg_per_neuron=agg_per_neuron,
                                                        agg_axis=agg_axis)
                curr_source_mapping_Ysmp_train = _return_curr(source_mapping_Ysmp_train,
                                                        curr_idx=ni,
                                                        agg_per_neuron=agg_per_neuron,
                                                        agg_axis=agg_axis)

                source_target_ds_train = _compute_metric_across_iters(Xsmp=curr_source_target_Xsmp_train,
                                                                Ysmp=curr_source_target_Ysmp_train,
                                                                metric=metric,
                                                                metric_name=metric_name,
                                                                mode=mode,
                                                                corrector=corrector)

                source_mapping_ds_train = _compute_metric_across_iters(Xsmp=curr_source_mapping_Xsmp_train,
                                                                 Ysmp=curr_source_mapping_Ysmp_train,
                                                                 metric=metric,
                                                                 metric_name=metric_name,
                                                                 mode=mode,
                                                             corrector=corrector)

            source_target_Xsmp = np.array(source_target_Xsmp) # in (n_reps, images, target_n_neu)
            source_target_Ysmp = np.array(source_target_Ysmp) # in (n_reps, images, target_n_neu)
            source_mapping_Xsmp = np.array(source_mapping_Xsmp) # in (n_reps, images, target_n_neu)
            source_mapping_Ysmp = np.array(source_mapping_Ysmp) # in (n_reps, images, target_n_neu)
            # subselect
            curr_source_target_Xsmp = _return_curr(source_target_Xsmp,
                                                   curr_idx=ni,
                                                   agg_per_neuron=agg_per_neuron,
                                                   agg_axis=agg_axis)
            curr_source_target_Ysmp = _return_curr(source_target_Ysmp,
                                                   curr_idx=ni,
                                                   agg_per_neuron=agg_per_neuron,
                                                   agg_axis=agg_axis)
            curr_source_mapping_Xsmp = _return_curr(source_mapping_Xsmp,
                                                    curr_idx=ni,
                                                    agg_per_neuron=agg_per_neuron,
                                                    agg_axis=agg_axis)
            curr_source_mapping_Ysmp = _return_curr(source_mapping_Ysmp,
                                                    curr_idx=ni,
                                                    agg_per_neuron=agg_per_neuron,
                                                    agg_axis=agg_axis)

            source_target_ds = _compute_metric_across_iters(Xsmp=curr_source_target_Xsmp,
                                                            Ysmp=curr_source_target_Ysmp,
                                                            metric=metric,
                                                            metric_name=metric_name,
                                                            mode=mode,
                                                            corrector=corrector)

            source_mapping_ds = _compute_metric_across_iters(Xsmp=curr_source_mapping_Xsmp,
                                                             Ysmp=curr_source_mapping_Ysmp,
                                                             metric=metric,
                                                             metric_name=metric_name,
                                                             mode=mode,
                                                             corrector=corrector)

            if report_train:
                numerator_train = source_target_ds_train
                if mode == 'spearman_brown_split_half_denominator':
                    source_mapping_ds_train = 2. * source_mapping_ds_train / (1. + source_mapping_ds_train)
                    ds_train = 2. * ds_train / (1. + ds_train)
                denominator_train = np.sqrt(source_mapping_ds_train * ds_train)
                ds_train = np.divide(numerator_train, denominator_train)

            numerator = source_target_ds
            if mode == 'spearman_brown_split_half_denominator':
                source_mapping_ds = 2. * source_mapping_ds / (1. + source_mapping_ds)
                ds = 2. * ds / (1. + ds)
            denominator = np.sqrt(source_mapping_ds * ds)
            # NOTE: for some target neurons this ratio will be NaN since the target test noise (ds above) can be a negative correlation (for some or all splits of the data)
            # In that case, we want this to be a NaN value since the assumptions that resulted in this quantity (e.g. transitivity of correlations) assume positive correlations
            ds = np.divide(numerator, denominator)

        if ('spearman_brown' in mode) and (mode != 'spearman_brown_split_half_denominator'):
            ds = 2. * ds / (1. + ds)
            if report_train:
                ds_train = 2. * ds_train / (1. + ds_train)

        if agg_per_neuron: # in this case ds is (n_reps,), so we add 1 to append the neurons dimension
            ds = np.expand_dims(ds, axis=agg_axis)
            if report_train:
                ds_train = np.expand_dims(ds_train, axis=agg_axis)
        # otherwise, if agg_per_neuron is False, ds is already (n_reps,) which is the axis (0) we want to concatenate across
        ds_agg.append(ds)
        if report_train:
            ds_agg_train.append(ds_train)

    ds_agg = np.concatenate(ds_agg, axis=agg_axis) # in (n_reps, target_n_neu) if agg_per_neuron is True, else in (n_reps) since the metric (e.g. RSA) is computed across images and neurons internally
    if report_train:
        ds_agg_train = np.concatenate(ds_agg_train, axis=agg_axis)
        return ds_agg, ds_agg_train
    else:
        return ds_agg
