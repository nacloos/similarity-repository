import numpy as np
import torch
import torch.nn as nn

from stir.CKA_pytorch import linear_CKA, kernel_CKA, WrapperCKA 
from stir.CKA_minibatch import MinibatchCKA

from functools import partial
from typing import Optional, Tuple, Union, List

SIM_METRICS = {
    'linear_CKA': partial(WrapperCKA, linear_CKA),
    'kernel_CKA': partial(WrapperCKA, kernel_CKA),
    'linear_CKA_batch': partial(MinibatchCKA)
} # instantiate a new object whenever needed


def forward_models(model, inp, layer_num=None):
    if isinstance(model, torch.nn.DataParallel):
        model_name = model.module.__class__.__name__
    else:
        model_name = model.__class__.__name__
    fake_relu = True
    if model_name == 'VGG' or \
        model_name == 'InceptionV3' or \
            'sparse' in model_name.lower():
        fake_relu = False
    return model(inp, with_latent=True, fake_relu=fake_relu, layer_num=layer_num)


class RelativeAdvLoss:
    """
    Loss of the form:
        CrossEntropyLoss + \
            beta * (Lpnorm of reps of initial image and 
            image supposed to have similar perception on reference models)
    """

    def __init__(self, args):
        self.args = args

    def __call__(self, original_image, matched_image, label, model):
        # op, rep1 = forward_models(model.model_generator.model if isinstance(model, hp.GeneratorWrapper) else model.model, original_image)
        op, rep1 = model(original_image, with_latent=True, with_image=False)
        # op_matched, rep2 = forward_models(model.model_generator.model if isinstance(model, hp.GeneratorWrapper) else model.model, matched_image)
        _, rep2 = model(matched_image, with_latent=True, with_image=False)
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')(op, label)
        self.perception_loss = torch.norm(rep1 - rep2, p=self.args.lpnorm_type, dim=1)
        self.normed_perception_loss = torch.div(torch.norm(rep1 - rep2, p=self.args.lpnorm_type, dim=1), 
            torch.norm(rep1, p=self.args.lpnorm_type, dim=1))
        self.loss = self.ce_loss + self.args.beta * self.normed_perception_loss

        torch.cuda.empty_cache()
        return self.loss, op

    def __repr__(self):
        return f'CE Loss: {torch.mean(self.ce_loss).item():.4f} + {self.args.beta} * {torch.mean(self.perception_loss).item():.4f}'


class ControversialStimuliLoss:
    """
    Loss of the form:
        LpLoss_model1 - beta * (LpLoss_model2)
    Minimizing this would make inputs similar on model1 but different on model2
    """

    def __init__(self, args, beta=0.1):
        self.args = args
        self.beta = beta

    def __call__(self, model1, model2, inp, targ1, targ2, layer_num=None):
        _, rep1 = model1(inp, with_latent=True)
        _, rep2 = model2(inp, with_latent=True)
        
        self.model1_loss_normed = torch.div(torch.norm(rep1 - targ1, p=self.args.lpnorm_type, dim=1), 
                                            torch.norm(targ1, p=self.args.lpnorm_type, dim=1))
        self.model2_loss_normed = torch.div(torch.norm(rep2 - targ2, p=self.args.lpnorm_type, dim=1), 
                                            torch.norm(targ2, p=self.args.lpnorm_type, dim=1))
        self.loss = self.model1_loss_normed - self.beta * self.model2_loss_normed

        rep1, rep2 = None, None
        torch.cuda.empty_cache()

        return self.loss, None

    def clear_cache(self) -> None:
        self.loss, self.model1_loss_normed, self.model2_loss_normed = None, None, None
        torch.cuda.empty_cache()

    def __repr__(self):
        return f'Total Loss: {torch.mean(self.loss).item():.4f} '\
            f'(model1: {torch.mean(self.model1_loss_normed).item():.4f}, '\
            f'model2: {torch.mean(self.model2_loss_normed).item():.4f})'


class LPNormLossSingleModel:
    
    def __init__(self, args):
        self.args = args
        self.model1_loss, self.model1_loss_normed = torch.zeros(1), torch.zeros(1)
    
    def __call__(self, model1, model2, inp, targ1, targ2, layer_num=None):
        _, rep1 = forward_models(model1, inp, layer_num)
        self.model1_loss_normed = torch.div(torch.norm(rep1 - targ1, p=self.args.lpnorm_type, dim=1), 
                                            torch.norm(targ1, p=self.args.lpnorm_type, dim=1))
        self.model1_loss = torch.norm(rep1 - targ1, p=self.args.lpnorm_type, dim=1)
        loss = self.model1_loss_normed

        rep1 = None
        torch.cuda.empty_cache()

        return loss, None
    
    def clear_cache(self) -> None:
        self.model1_loss, self.model1_loss_normed = None, None
        torch.cuda.empty_cache()

    def __repr__(self):
        return f'Model1 Loss: {torch.mean(self.model1_loss)} ({torch.mean(self.model1_loss_normed)})'


class LPNormLoss:

    def __init__(self, args):
        self.args = args

    def __call__(self, model1, model2, inp, targ1, targ2):
        _, rep1 = forward_models(model1, inp)
        _, rep2 = forward_models(model2, inp)
        self.model1_loss_normed = torch.div(torch.norm(rep1 - targ1, p=self.args.lpnorm_type, dim=1), 
                                            torch.norm(targ1, p=self.args.lpnorm_type, dim=1))
        self.model2_loss_normed = torch.div(torch.norm(rep2 - targ2, p=self.args.lpnorm_type, dim=1), 
                                            torch.norm(targ2, p=self.args.lpnorm_type, dim=1))
        self.model1_loss = torch.norm(rep1 - targ1, p=self.args.lpnorm_type, dim=1)
        self.model2_loss = torch.norm(rep2 - targ2, p=self.args.lpnorm_type, dim=1)
        if self.args.alpha is not None:
            alpha = self.args.alpha
        else:
            alpha = torch.mean(self.model1_loss_normed).item() / torch.mean(self.model2_loss_normed).item()
        loss = (1./alpha) * self.model1_loss_normed - self.model2_loss_normed
        # loss = (1./alpha) * self.model1_loss - self.model2_loss

        rep1, rep2 = None, None
        torch.cuda.empty_cache()

        return loss, None

    def __repr__(self):
        return f'Model1 Loss: {torch.mean(self.model1_loss)} ({torch.mean(self.model1_loss_normed)}), Model2 Loss {torch.mean(self.model2_loss)} ({torch.mean(self.model2_loss_normed)})'


class SimLoss:

    def __init__(self, args) -> None:
        self.args = args
        self.CKA_func = SIM_METRICS[args.sim_metric]
        self.cka_val = 0.
    
    def set_original_inputs(self, original_inputs) -> None:
        pass

    def __call__(self, model1, model2, inp, targ1, targ2, layer_num=None) -> torch.Tensor:
        '''
        model1: instance of nn.Module
        model2: instance of nn.Module
        inp: batch of input samples, should be normalized 
                accordingly before being passed to this function
                For adv attacks, `inp` should have 
                `requires_grad=True` before being passed 
                into this function
        targ1, targ2: placeholders so all custom losses have the 
                        same call function
        '''

        _, rep1 = model1(inp, with_latent=True)
        _, rep2 = model2(inp, with_latent=True)

        cka = self.CKA_func(rep1, rep2)
        self.cka_val = cka.item()

        rep1, rep2 = None, None
        torch.cuda.empty_cache()

        return cka, None # this is a scalar

    def __repr__(self) -> str:
        return f'CKA score: {self.cka_val:.5f}'


class SimLossConstant:
    """
    Useful when attacking CKA while keeping CKA on model1/model2 constant
    """

    def __init__(self, args, constant_model='model1') -> None:
        self.args = args
        self.CKA_func = SIM_METRICS[args.sim_metric]
        self.cka_val, self.cka_val_wrtm1, self.cka_val_wrtm2 = 0., 0., 0.
        self.original_inputs = None
        self.constant_model = constant_model
    
    def set_original_inputs(self, original_inputs) -> None:
        self.original_inputs = original_inputs
    
    def __call__(self, model1, model2, inp, targ1, targ2, layer_num=None) -> torch.Tensor:
        '''
        model1: instance of nn.Module
        model2: instance of nn.Module
        inp: batch of input samples, should be normalized 
                accordingly before being passed to this function
                For adv attacks, `inp` should have 
                `requires_grad=True` before being passed 
                into this function
        targ1, targ2: placeholders so all custom losses have the 
                        same call function
        '''
        assert self.original_inputs is not None, 'Must call set_original_inputs before running'
        
        _, rep1 = model1(inp, with_latent=True)
        _, rep2 = model2(inp, with_latent=True)

        _, rep_og1 = model1(self.original_inputs, with_latent=True)
        _, rep_og2 = model2(self.original_inputs, with_latent=True)
        rep_og1, rep_og2 = rep_og1.detach(), rep_og2.detach()

        cka = self.CKA_func(rep1, rep2)
        cka_wrtm1 = self.CKA_func(rep1, rep_og1)
        cka_wrtm2 = self.CKA_func(rep2, rep_og2)

        if self.constant_model == 'model1':
            final_loss = cka - cka_wrtm1 + cka_wrtm2
        elif self.constant_model == 'model2':
            final_loss = cka - cka_wrtm2 + cka_wrtm1
        else:
            final_loss = cka + cka_wrtm2 + cka_wrtm1
        
        self.cka_val = cka.item()
        self.cka_val_wrtm1 = cka_wrtm1.item()
        self.cka_val_wrtm2 = cka_wrtm2.item()

        return final_loss, None # this is a scalar

    def __repr__(self) -> str:
        return f'Keeping {self.constant_model} constant,'\
               f' CKA score: {self.cka_val:.5f}, m1: {self.cka_val_wrtm1:.5f}, m2: {self.cka_val_wrtm2:.5f}'

