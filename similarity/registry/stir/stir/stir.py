'''
A relaxation of inverted_rep_divergence, we calculate CKA on m2(X) and 
m2(X') where X' are inverted reps of m1(X), ie m1(X) \approx m1(X').
'''

from typing import Iterable, Union, Dict, Optional
import torch

from stir.attack.losses import SIM_METRICS, LPNormLossSingleModel
from stir.attack.attacker import AttackerModel
import stir.helper as hp
import stir.model.tools.helpers as helpers


def get_seed_images(seed, shape, verbose, inputs=None):
    if verbose:
        print (f'From get_seed_images, seed = {seed}')
    if seed == 'super-noise-same':
        if verbose:    print('=> Seeds: Random super-noise, same for all images')
        init_seed_images = torch.randn(*shape[1:], dtype=torch.float)
        init_seed_images = torch.stack([init_seed_images] * shape[0])
    elif seed == 'completely-random':
        if verbose:    print('=> Seeds: Completely random images')
        init_seed_images = torch.randint(0, 255, size=shape[1:], dtype=torch.float)/255.
        init_seed_images = torch.stack([init_seed_images] * shape[0])
    elif seed == 'super-noise':
        if verbose:    print('=> Seeds: Random super-noise')
        init_seed_images = torch.randn(*shape, dtype=torch.float)
    elif seed == 'white':
        if verbose:    print('=> Seeds: All white')
        init_seed_images = torch.ones(*shape, dtype=torch.float)
    elif seed == 'black':
        if verbose:    print('=> Seeds: All black')
        init_seed_images = torch.zeros(*shape, dtype=torch.float)
    elif seed == 'all_types':
        if verbose:    print('=> Seeds: Using all types of seeds')
        all_seeds = ['super-noise-same', 'white', 'black']
        if shape[0] % len(all_seeds) == 0:
            num_samples = [int(shape[0] / len(all_seeds))] * len(all_seeds)
        else:
            num_samples = [int(shape[0] / len(all_seeds))] * len(all_seeds)
            num_samples[-1] += int(shape[0] % len(all_seeds))
        assert sum(num_samples) == shape[0], (num_samples, shape[0])
        init_seed_images = None
        for s, num in zip(all_seeds, num_samples):
            init_seed_images = get_seed_images(s, (num, *shape[1:]), verbose=False) \
                if init_seed_images is None else \
                    torch.cat((init_seed_images, get_seed_images(s, (num, *shape[1:]), verbose=False)))
    elif seed == 'mixed_inputs':
        assert inputs is not None, f'Inputs must be provided for seed = {seed}'
        init_seed_images = torch.flip(inputs, (0,))
    else:
        raise ValueError(f'Seed {seed} not supported!')
    return init_seed_images


class STIRResult:
    def __init__(self, stir_m1m2, stir_m2m1, rsm):
        self.m1m2 = stir_m1m2
        self.m2m1 = stir_m2m1
        self.rsm = rsm


def STIR(model1: torch.nn.Module, 
         model2: torch.nn.Module, 
         normalizer1: torch.nn.Module, 
         normalizer2: torch.nn.Module, 
         inputs: Union[tuple, torch.Tensor], 
         devices: list=[0], 
         ve_kwargs: Optional[Dict]=None, 
         norm_type: int=2., 
         seed: str='super-noise', 
         verbose=False, 
         sim_metric: str='linear_CKA', 
         no_opt=False, 
         layer1_num=None, 
         layer2_num=None):

    if isinstance(inputs, tuple):
        loader, total_imgs = inputs
        if not isinstance(loader, helpers.DataPrefetcher):
            loader = helpers.DataPrefetcher(loader, 
                device=torch.device(f'cuda:{devices[0]}'))
    else:
        loader, total_imgs = \
            [(inputs, torch.arange(len(inputs)))], len(inputs) # to make it an iterable
    
    model1, model2 = AttackerModel(model1, normalizer1), AttackerModel(model2, normalizer2)
    model1, model2 = model1.to(torch.device(f'cuda:{devices[0]}')), model2.to(torch.device(f'cuda:{devices[0]}'))
    m2m1, _ = _stir(model1, model2, loader, total_imgs, devices, ve_kwargs, 
                    norm_type, seed, verbose, sim_metric, no_opt, 
                    layer1_num, layer2_num)
    m1m2, rsm = _stir(model2, model1, loader, total_imgs, devices, ve_kwargs, 
                      norm_type, seed, verbose, sim_metric, no_opt, 
                      layer2_num, layer1_num)
    return STIRResult(m1m2, m2m1, rsm)


def _stir(model1: AttackerModel, 
         model2: AttackerModel, 
         loader: Iterable, 
         total_imgs: int,
         devices: list=[0], 
         ve_kwargs: Optional[Dict]=None, 
         norm_type: int=2., 
         seed: str='super-noise', 
         verbose=True, 
         sim_metric: str='linear_CKA', 
         no_opt=False, 
         layer1_num=None, 
         layer2_num=None):
    '''
    Given two models model1 and model2 (both instances of attacker.Attacker),
    it returns STIR(model2 | model1)

    This is done by generating a set of images (X, X') that are perceived similarly 
    by model1 and then giving it to model2 and calculating similarity 
    between m2(X) and m2(X').

    inputs: either a set of images (torch.Tensor) or a tuple of (loader, total_imgs)
    loader: loads images on which to calculate divergences
    devices: GPU device indices
    no_opt: if True, returns STIR scores on just the seeds -- only use for sanity check
    seed: starting point for representation inversion
    norm_type: norm used to generate inverted reresentations
    '''

    if ve_kwargs is None:
        dummy_args = hp.DummyArgs(('lpnorm_type', norm_type, float)) # LpNorm for the representation
        ve_kwargs = {
            'custom_loss': LPNormLossSingleModel(dummy_args),
            'constraint': 'unconstrained',
            'eps': 1000, # put whatever, threat model is unconstrained, so this does not matter
            'step_size': 0.5,
            'iterations': 1000,
            'targeted': True,
            'do_tqdm': verbose,
            'should_normalize': True,
            'use_best': True
        }
    ve_kwargs['layer_num'] = layer1_num

    n_samples_per_iter = None
    ## create two objects, one to track the underlying RSM, one to track STIR
    cka_og, cka_stir = SIM_METRICS[sim_metric](), SIM_METRICS[sim_metric]()
    for index, tup in enumerate(loader):
        images, _ = tup
        n_samples_this_iter = len(images)
        if n_samples_per_iter is None:  n_samples_per_iter = len(images)

        (_, images_repr1), _ = model1(images, layer_num=layer1_num, 
            with_latent=True if layer1_num is None else False)
        images_repr1 = images_repr1.detach()
        (_, images_repr2), _ = model2(images, layer_num=layer2_num, 
            with_latent=True if layer2_num is None else False)
        images_repr2 = images_repr2.detach()
        
        cka_og(images_repr1, images_repr2)

        seed_images = get_seed_images(seed, images.shape, verbose, inputs=images).to(
            torch.device(f'cuda:{devices[0]}'))

        (_, seed_reps_1), _ = model1(seed_images, layer_num=layer1_num, 
            with_latent=True if layer1_num is None else False)
        seed_reps_1 = seed_reps_1.detach()
        (_, seed_reps_2), _ = model2(seed_images, layer_num=layer2_num, 
            with_latent=True if layer2_num is None else False)
        seed_reps_2 = seed_reps_2.detach()

        if verbose:
            print ('Initial seed and target rep distance on model1: '
                   f'{torch.mean(torch.norm(images_repr1 - seed_reps_1, p=norm_type, dim=1))}')
            print ('Initial seed and target rep distance on model2: '
                   f'{torch.mean(torch.norm(images_repr2 - seed_reps_2, p=norm_type, dim=1))}')

        if no_opt:
            images_matched = seed_images
        else:
            (_, images_matched), _ = model1(
                inp=seed_images,
                target=images_repr1,
                make_adv=True,
                with_image=True,
                **ve_kwargs) # these images are not normalized
            images_matched = images_matched.detach()

        seed_reps_1, seed_reps_2, seed_images, images_repr1, images_repr2 = \
            None, None, None, None, None
        torch.cuda.empty_cache()

        _, rep_x = model2(images, with_latent=True if layer2_num is None else False, 
            with_image=False, make_adv=False, layer_num=layer2_num)
        rep_x = rep_x.detach()
        _, rep_y = model2(images_matched, with_latent=True if layer2_num is None else False, 
            with_image=False, make_adv=False, layer_num=layer2_num)
        rep_y = rep_y.detach()
        
        cka_stir(rep_x, rep_y)

        rep_x, rep_y, images, images_matched = None, None, None, None
        torch.cuda.empty_cache()

        if index*n_samples_per_iter + n_samples_this_iter == total_imgs:
            break

    ve_kwargs['custom_loss'].clear_cache()

    return cka_stir.value(), cka_og.value()
