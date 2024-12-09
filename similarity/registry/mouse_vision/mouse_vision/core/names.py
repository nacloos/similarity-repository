from mouse_vision.core.utils import dict_to_str


def filename_constructor(
    source_map_kwargs=None,
    fit_per_target_unit=False,
    num_source_units=None,
    source_unit_percentage=None,
    train_frac=None,
    num_train_test_splits=None,
    n_ss_iter=None,
    n_iter=None,
    n_ss_imgs=None,
    equalize_source_units=False,
    metric_name=None,
    correction="spearman_brown_split_half",
    splithalf_r_thresh=None,
    names="",
    append_file_ext=True,
    shorten_name=False,
    shorten_perc_name=False,
):

    file_nm = dict_to_str(names) if isinstance(names, dict) else "{}".format(names)
    if metric_name is not None:
        file_nm += "_metric{}".format(metric_name)
    if equalize_source_units:
        file_nm += "_equalize_source_units"
    if n_ss_iter is not None:
        file_nm += "_nssiter{}".format(n_ss_iter)
    if n_iter is not None:
        file_nm += "_niter{}".format(n_iter)
    if n_ss_imgs is not None:
        file_nm += "_nssimgs{}".format(n_ss_imgs)
    if splithalf_r_thresh is not None:
        file_nm += "_sphr{}".format(splithalf_r_thresh)

    if source_map_kwargs is not None:
        file_nm += "_{}".format(dict_to_str(source_map_kwargs))
        if shorten_name:
            if num_source_units is not None:
                file_nm += "_numsourceunits{}".format(num_source_units)
            if source_unit_percentage is not None:
                if shorten_perc_name:
                    file_nm += "_spe{:.2f}".format(source_unit_percentage)
                else:
                    file_nm += "_sourceunitperc{:.3f}".format(source_unit_percentage)
        else:
            file_nm += "_fitpertarget{}".format(fit_per_target_unit)
            file_nm += "_numsourceunits{}".format(num_source_units)
            if source_unit_percentage is not None:
                if shorten_perc_name:
                    file_nm += "_spe{:.2f}".format(source_unit_percentage)
                else:
                    file_nm += "_sourceunitperc{:.3f}".format(source_unit_percentage)

    if train_frac is not None:
        file_nm += "_trainfrac{}".format(train_frac)
    if num_train_test_splits is not None:
        file_nm += "_numsplits{}".format(num_train_test_splits)

    if correction != "spearman_brown_split_half":
        file_nm += "_correction{}".format(correction)

    if shorten_name:
        file_nm = file_nm.replace("_", "")

    if append_file_ext:
        file_nm += ".pkl"

    return file_nm
