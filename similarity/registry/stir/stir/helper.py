from ast import excepthandler
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from enum import Enum
import math, os
import sys, pickle, functools

import joblib, pickle
from sklearn.metrics import accuracy_score
from stir.attack.losses import RelativeAdvLoss, LPNormLossSingleModel

RESULTS_FOLDER_NAME = 'results'

# layers that can be reset
MODEL_TO_LAYER_RANGE = {
    'ResNet': range(1,7),
    'DenseNet': range(1,10),
    'VGG': range(1,46),
}

# layers that can be unfrozen
MODEL_TO_LAYER_IDX = {
    'ResNet': range(1,7),
    'DenseNet': range(1,11),
    'VGG': range(1,45),
}

INPUT_SHAPES = {
    'stl10': 96,
    'cifar10': 32,
    'cifar': 32,
    'cifar100': 32,
    'imagenet': 224
}

DATASET_TO_INPUT_SHAPE = lambda x: INPUT_SHAPES[x.split('_', 1)[0]]

DATASET_TO_MEAN_STD = {
    'cifar': (torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2023, 0.1994, 0.2010])),
    'cifar10': (torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2023, 0.1994, 0.2010])),
    'cifar100': (torch.tensor([0.5071, 0.4865, 0.4409]), torch.tensor([0.2673, 0.2564, 0.2762])),
    'stl10': (torch.tensor([0.4467, 0.4398, 0.4066]), torch.tensor([0.2603, 0.2566, 0.2713])),
    'imagenet': (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
}


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def recursive_create_dir(target_dir):
    directories = ['/'] + target_dir.split('/')[1:-1]
    for i in range(1, len(directories) + 1):
        yield os.path.join(*directories[:i])


def list_as_ints(arg):
    return [int(x) if x.lower() != 'none' else None for x in arg.split(',')]

def list_as_strs(arg):
    return [str(x) if x.lower() != 'none' else None for x in arg.split(',')]

def list_as_floats(arg):
    return [float(x) if x.lower() != 'none' else None for x in arg.split(',')]

def eval_model(model, data_loader, devices):
    model = model.eval()
    preds, true = None, None
    for image, label in data_loader:
        if isinstance(image, tuple) or isinstance(image, list): # (target, result)
            image = image[1]
        try:
            image = image.to(devices[0])
        except:
            print (f'Devices ({devices}) not valid!')
        out = model(image, with_image=False)
        out = torch.argmax(out, 1)
        preds = out.detach().cpu().numpy() if preds is None else \
            np.concatenate((preds, out.detach().cpu().numpy()))
        true = label.cpu().numpy() if true is None else np.concatenate((true, label.cpu().numpy()))

        image, out = None, None
        torch.cuda.empty_cache()
    
    return accuracy_score(true, preds)

def get_wiki_link(figure_path, size=500):
    """Returns the link to an image in wiki format"""
    server_path = figure_path.replace('results', SERVER_PROJECT_PATH)
    wiki_link = out.web_attachment(server_path, size=size)
    return wiki_link

def get_params(model, num_unfrozen):
    model_parameters = list(model.parameters())
    for param in model_parameters:
        param.requires_grad = False
    params_to_update = []
    if num_unfrozen is None:    num_unfrozen = len(model_parameters)
    for i in range(1, num_unfrozen + 1):
        # model_parameters[-2-i].requires_grad = True
        model_parameters[-i].requires_grad = True
        params_to_update.append(model_parameters[-i])
    
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return params_to_update, active_params, total_params

class LpNormLoss:

    def __init__(self, norm='fro'):
        self.norm = norm

    def __call__(self, output, target):
        return torch.norm(output - target, dim=1, p=self.norm)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return 'Lp norm distance'


def consolidate_loss(inp, target, model, adv, loss_criterion, is_train, attack_kwargs, extra_kwargs={}, devices=[0]):
    """
    inp: could be just an image tensor, or a tuple of image tensors indicating (target, result)
    target: output label of images in inp
    model: pytorch model
    adv: bool, adversarial training or not
    loss_criterion: CrossEntropyLoss or custom loss
    attack_kwargs: kwargs for adv attack
    extra_kwargs: kwargs to be passed to model call. Useful for contrastive adv training
        inp: seed images used to generated images with similar perception on reference models,
        models: reference models,
        target: target_reps in penultimate layer,
        with_image: image generated to match the perception
    """
    extra = []
    if isinstance(inp, tuple) or isinstance(inp, list):
        if isinstance(loss_criterion, LpNormLoss):
            _, target_image, _, result_image = inp
            # target_image, result_image = inp
            target_image, result_image = target_image.to(f'cuda:{devices[0]}'), \
                result_image.to(f'cuda:{devices[0]}')
            result_image.requires_grad = True
            output, result_rep = model(result_image, with_latent=True, with_image=False)
            # result_rep = model(result_image, with_image=False)

            # model = model.eval() # we don't want to create computation graph for target
            _, target_rep = model(target_image, with_latent=True, with_image=False)
            # target_rep = model(target_image, with_image=False)
            # print (target_rep, target_rep.shape)
            
            loss = loss_criterion(result_rep, target_rep)
            return result_image, loss, output, result_rep
        else:
            inp = inp[-1]
            inp = inp.to(f'cuda:{devices[0]}')

    if isinstance(loss_criterion, RelativeAdvLoss):
        assert is_train
        ## Handle extra_kwargs for relative adv training here.
        # For contrastive adversarial training need to pass extra_kwargs 
        # that will create inputs same for model but different for reference_models
        # These inputs must then be used in loss_criterion for training. 
        # seed_images and target_reprs must be created here since they are not 
        # in extra_kwargs, which was generated before training started.
        target_reprs = []
        for m in extra_kwargs['models']:
            (_, images_repr), _ = m(inp, with_latent=True)
            target_reprs.append(images_repr.detach())
        seed_images = inp + torch.normal(mean=0., std=0.00001, size=inp.shape).to(f'cuda:{devices[0]}')
        # generate extra inputs
        (_, final_inp), _ = model(inp=seed_images, target=target_reprs, make_adv=adv,
                                  **attack_kwargs, **extra_kwargs)
        # add this to CE Loss and 
        loss, output = loss_criterion(inp, final_inp, target, model)
    else:
        if adv:
            if isinstance(attack_kwargs['custom_loss'], LPNormLossSingleModel) or \
                isinstance(attack_kwargs['custom_loss'], LPNormLossSingleModel):
                noise = torch.ones_like(inp)
                torch.normal(mean=0., std=0.00001, size=noise.shape, out=noise)
                seed_images = inp + noise
                (output_original, latent_original), _ = model(inp, with_latent=True)
                (_, latent_noisy), _ = model(seed_images, with_latent=True)
                latent_original, latent_noisy = latent_original.detach(), latent_noisy.detach()
                print (f'Initial perception dist: {torch.mean(torch.norm(latent_noisy - latent_original, dim=1))}')
                ((output, latent), final_inp), _ = model(
                    seed_images,
                    target=latent_original, 
                    make_adv=True,
                    with_image=True,
                    with_latent=True,
                    do_tqdm=False,
                    **attack_kwargs)
                seed_images, noise, latent_noisy = None, None, None
                print (f'Final perception dist: {torch.mean(torch.norm(latent - latent_original, dim=1))}')
            else:
                ((output, latent), final_inp), _ = model(inp, target=target, make_adv=adv, 
                                        with_image=True, with_latent=True,
                                        **attack_kwargs) # extra kwargs are only for visual expl generation
                output_original, latent_original = model(inp, target=target, make_adv=False, 
                                        with_image=False, with_latent=True)

            output_original, latent_original, latent, final_inp = \
                output_original.detach(), latent_original.detach(), \
                    latent.detach(), final_inp.detach()
            torch.cuda.empty_cache()
            extra.extend([latent, latent_original])
        else:
            output, final_inp = model(inp, target=target, make_adv=adv, with_image=True,
                                    **attack_kwargs) # extra kwargs are only for visual expl generation
            output_original = output.clone().detach()
        
        loss = loss_criterion(output, target)

    return (inp, loss, output_original, output, final_inp, *extra)

def get_epsilons(dataset):
    if dataset == 'imagenet':
        return np.concatenate((np.arange(0, 5.1, 0.1), np.arange(5, 50, 10)))
    if dataset == 'cifar':
        return np.arange(0.0, 1.51, 0.05)

def create_batches(iterables, batch_size):
    total_size = len(iterables[0])
    num_batches = math.ceil(total_size/batch_size)
    for batch_idx in range(num_batches):
        yield (x[batch_idx * batch_size:(batch_idx + 1) * batch_size] for x in iterables)

def save_objects(adversarial_objects, adversarial_image_ids, model_name, attack_name, out_dir, epsilon):
    out.create_dir(f"{out_dir}")
    out.create_dir(f"{out_dir}/{model_name}")
    out.create_dir(f"{out_dir}/{model_name}/{attack_name}")
    out.create_dir(f"{out_dir}/{model_name}/{attack_name}/epsilon_{epsilon:.2f}")
    for image, image_id in zip(adversarial_objects, adversarial_image_ids):
        assert image_id == image.image_id
        filename = f'{int(image_id)}.pkl'
        with open(f'{out_dir}/{model_name}/{attack_name}/epsilon_{epsilon:.2f}/{filename}', 'wb') as handle:
            pickle.dump(image, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_objects(adversarial_image_ids, model_name, attack_name, out_dir, epsilon):
    base_dir = f"{out_dir}/{model_name}/{attack_name}/epsilon_{epsilon:.2f}"
    adv_objects = []
    for image_id in adversarial_image_ids:
        filename = f'{base_dir}/{int(image_id)}.pkl'
        adv_objects.append(joblib.load(filename))
    return adv_objects


def get_random_images(data_loader, num_images):
    result = None
    for image, label in data_loader:
        if isinstance(image, tuple) or isinstance(image, list): # (target, result)
            image = image[1]
        result = image if result is None else torch.cat((result, image))

        if len(result) >= num_images:
            break
    
    return result[:num_images,:,:,:] if len(result.shape) == 4 else result[:num_images,:,:] # accounting for B/W images

def get_classes_names(dataset, data_path):
    if dataset == 'imagenet':
        dset = datasets.ImageNet(data_path, split='val')
        ordered_class_labels = []
        for idx, x in enumerate(dset.classes):
            all_names = []
            for i in range(len(x)):
                all_names.append('_'.join(x[i].split()))
            if idx == 134:
                all_names.append('bird')
            elif idx == 517:
                all_names.append('machine')
            ordered_class_labels.append('-'.join(all_names))
        return ordered_class_labels
    elif dataset == 'cifar' or dataset == 'cifar10':
        dset = datasets.CIFAR10(data_path, train=False)
        return dset.classes
    elif dataset == 'cifar100':
        dset = datasets.CIFAR100(data_path, train=False)
        return dset.classes
    else:
        raise ValueError(f'Dataset {dataset} not recognized')


def norm_batch_images(images_batch, p=2):
    final = torch.tensor([])
    for img in images_batch:
        final = torch.cat((final, torch.tensor([torch.norm(img, p=p)])))
    return final

def consolidate_attack_kwargs(args, adv_criterion, random_restarts, eps, is_train, devices):
    """
    For adversarial training, generates args for the attacker.
    kwargs to args might be different based on mode being train or val.
    """
    if args.separate_attack_args:
        if is_train:
            attack_kwargs = {
                'constraint': args.constraint_train,
                'eps': eps,
                'step_size': args.attack_lr_train,
                'iterations': args.attack_steps_train,
                'random_start': args.random_start,
                'custom_loss': adv_criterion,
                'random_restarts': random_restarts,
                'use_best': bool(args.use_best),
                'devices': devices
            }
        else:
            attack_kwargs = {
                'constraint': args.constraint_test,
                'eps': eps,
                'step_size': args.attack_lr_test,
                'iterations': args.attack_steps_test,
                'random_start': args.random_start,
                'custom_loss': None, # for evaluation, just use CrossEntropyLoss
                'random_restarts': random_restarts,
                'use_best': bool(args.use_best),
                'devices': devices
            }
    else:
        attack_kwargs = {
                'constraint': args.constraint,
                'eps': eps,
                'step_size': args.attack_lr,
                'iterations': args.attack_steps,
                'random_start': args.random_start,
                'custom_loss': adv_criterion,
                'random_restarts': random_restarts,
                'use_best': bool(args.use_best),
                'devices': devices
            }
    return attack_kwargs

def add_extra_args(args, mapping):
    """
    args: Namespace
        argparse Namespace that has already been parsed
    mapping: dict
        key value to add to args
    """
    for k, v in mapping.items():
        args.__setattr__(k, v)
    return args

## https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def re_init_layers(model, layer_idx):
    """
    Given a model, re-initializes all layers including and after layer_idx
    """

    assert layer_idx in MODEL_TO_LAYER_IDX[model.model.__class__.__name__]

    count = None
    param_strs = []
    if model.model.__class__.__name__ == 'ResNet':
        for idx, _ in model.named_parameters():
            if '.'.join(idx.split('.')[:-1]) not in param_strs:
                param_strs.append('.'.join(idx.split('.')[:-1]))
        for p in param_strs:
            
            if 'model.bn1' in p or 'model.conv1' in p:
                count = 1
            elif 'model.layer' in p:
                count = int(p.split('.')[1][-1]) + 1
            elif 'model.linear' in p:
                count = 6
            
            if count >= layer_idx:
                rgetattr(model, p).reset_parameters()
                print (f'Layer {p} reset!')
            
def add_fc_layers(model, num_layers):
    """
    model (nn.Module): model with one final layer
    num_layers: number of fully connected layers to insert before the prediction layer
    """
    final_layer_name = list(model.named_modules())[-1][0]
    final_layer = model.__getattr__(final_layer_name)
    fc_layers = []
    for _ in range(num_layers - 1):
        fc_layers.append(nn.Linear(final_layer.in_features, final_layer.in_features))
    fc_layers.append(nn.Linear(final_layer.in_features, final_layer.out_features))
    model.__setattr__(final_layer_name, nn.Sequential(*fc_layers))


def accuracy_by_class(predictions, ground_truth, classes):
    class_wise_acc = []
    for i, _ in enumerate(classes):
        class_wise_acc.append(
            accuracy_score(predictions[ground_truth == i], ground_truth[ground_truth == i]))
    return class_wise_acc


def get_model_name(path):
    m1_info = path.split('checkpoints/')[1].split('/')
    mname1 = m1_info[1]
    mtype = m1_info[2]
    seed1 = m1_info[-1].split('checkpoint_')[1].split('.pt.best')[0]
    return f'{mtype}_{mname1}_{seed1}'


class GeneratorWrapper(nn.Module):
    """
    This is a wrapper to make a genertor act like a function
    """
    def __init__(self, model_generator):
        super(GeneratorWrapper, self).__init__()
        self.model_generator = model_generator

    def forward(self, *input_args, **input_kwargs):
        for op in self.model_generator(*input_args, **input_kwargs):
            return op

class ImageNetDatasetWrapper(datasets.ImageNet):

    def __init__(self, root, train, download, transform=None, **custom_class_args):
        super(ImageNetDatasetWrapper, self).__init__(root=root, split='train' if train else 'val', 
            transform=transform, **custom_class_args)

class DummyArgs:
    '''
    Placeholder for an argparse type object

    elements can be accessed through
    '''
    def __init__(self, *args):
        for arg in args:
            name, value, casttype = arg
            self.__setattr__(name, casttype(value))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1.):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
