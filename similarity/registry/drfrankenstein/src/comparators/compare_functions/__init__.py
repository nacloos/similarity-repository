from .cca import cca
from .cca_torch import cca as cca_torch
from .cka import cka
from .cka_torch import cka as cka_torch
from .correlation import correlation
from .l2 import l2
from .lr import lr
from .lr_torch import lr as lr_torch
# from .ls_orth import ls_orth
# from .ls_sum import ls_sum
# from .ps_inv import ps_inv
# from .tsne import tsne


def get_comparator_function(str_comparator):
    str_comparator = str_comparator.lower()
    dispatcher = {
        'cka': cka,
        'cca': cca,
        'l2': l2,
        'ps_inv': ps_inv,
        'ls_orth': ls_orth,
        'corr': correlation,
        'lr': lr,
        'ls_sum': ls_sum,
        'cca_torch': cca_torch,
        'lr_torch': lr_torch,
        'cka_torch': cka_torch,
    }

    if str_comparator not in dispatcher:
        raise ValueError('{} is unknown comparator.'.format(str_comparator))

    return dispatcher[str_comparator]
