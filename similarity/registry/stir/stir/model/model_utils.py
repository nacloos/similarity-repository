import torch as ch
from torch import nn
import dill
import os
from stir.attack.attacker import AttackerModel
import stir.model.tools.helpers as helpers

class FeatureExtractor(ch.nn.Module):
    '''
    Tool for extracting layers from models.

    Args:
        submod (torch.nn.Module): model to extract activations from
        layers (list of functions): list of functions where each function,
            when applied to submod, returns a desired layer. For example, one
            function could be `lambda model: model.layer1`.

    Returns:
        A model whose forward function returns the activations from the layers
            corresponding to the functions in `layers` (in the order that the
            functions were passed in the list).
    '''
    def __init__(self, submod, layers):
        # layers must be in order
        super(FeatureExtractor, self).__init__()
        self.submod = submod
        self.layers = layers
        self.n = 0

        for layer_func in layers:
            layer = layer_func(self.submod)
            def hook(module, _, output):
                module.register_buffer('activations', output)

            layer.register_forward_hook(hook)

    def forward(self, *args, **kwargs):
        """
        """
        # self.layer_outputs = {}
        out = self.submod(*args, **kwargs)
        activs = [layer_fn(self.submod).activations for layer_fn in self.layers]
        return [out] + activs

class DummyModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        return self.model(x)

def make_and_restore_model(*_, arch, dataset, target_dataset=None, resume_path=None, 
         source_dataset_shape=32, target_dataset_shape=32,
         parallel=False, pytorch_pretrained=False, add_custom_forward=False,
         trades=False, devices=[0], **extra_model_kwargs):
    '''
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py]) -- source dataset
        resume_path (str): optional path to checkpoint saved with the 
            robustness library (ignored if ``arch`` is not a string)
        not a string
        parallel (bool): if True, wrap the model in a DataParallel 
            (defaults to False)
        pytorch_pretrained (bool): if True, try to load a standard-trained 
            checkpoint from the torchvision library (throw error if failed)
        add_custom_forward (bool): ignored unless arch is an instance of
            nn.Module (and not a string). Normally, architectures should have a
            forward() function which accepts arguments ``with_latent``,
            ``fake_relu``, and ``no_relu`` to allow for adversarial manipulation
            (see `here`<https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures>
            for more info). If this argument is True, then these options will
            not be passed to forward(). (Useful if you just want to train a
            model and don't care about these arguments, and are passing in an
            arch that you don't want to edit forward() for, e.g.  a pretrained model)
        devices:
            passed as argument to torch.nn.DataParallel's visible_devices arg
    Returns: 
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    '''
    if (not isinstance(arch, str)) and add_custom_forward:
        arch = DummyModel(arch)

    if target_dataset is None:
        target_dataset = dataset

    classifier_model = dataset.get_model(arch, pytorch_pretrained, **extra_model_kwargs) if \
                            isinstance(arch, str) else arch

    model = AttackerModel(classifier_model, 
        helpers.InputNormalize(dataset.mean, dataset.std))

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = ch.load(resume_path, pickle_module=dill, map_location=f'cuda:{devices[0]}')
        
        if trades:
            if 'SimCLR' in resume_path or 'RoCL' in resume_path:
                named_components = resume_path.split('/')[-1].split('_linear_')[1].split('.')
                base_model_path = 'checkpoint.{}.{}'.format(*named_components[1:]) if 'linear_nobn' in resume_path else \
                    'checkpoint.{}.{}.{}'.format(named_components[1], named_components[0], named_components[2])
                base_model_path = f'{"/".join(resume_path.split("/")[:-1])}/{base_model_path}'    
                assert os.path.exists(base_model_path)
                base_checkpoint = ch.load(base_model_path, pickle_module=dill, map_location=f'cuda:{devices[0]}')
                base_checkpoint = {k[len('module.'):] if 'module.' in k else k:v for k, v in base_checkpoint.items() \
                    if k in model.model.state_dict() or k[len('module.'):] in model.model.state_dict()}
                if model.model.__class__.__name__ == 'ResNet':
                    if source_dataset_shape > 32:
                        in_ftrs = model.model.fc.in_features
                        model.model.fc = nn.Linear(in_ftrs, target_dataset.num_classes)
                    else:
                        in_ftrs = int(512 * ((target_dataset_shape ** 2) / (source_dataset_shape ** 2)))
                        model.model.linear = nn.Linear(in_ftrs, target_dataset.num_classes)
                    for k, v in checkpoint.items():
                        assert 'bias' in k or 'weight' in k
                        if source_dataset_shape > 32:
                            base_checkpoint[f'fc.{k.split(".")[-1]}'] = v
                        else:
                            base_checkpoint[f'linear.{k.split(".")[-1]}'] = v
                    model.model.load_state_dict(base_checkpoint)
                elif model.model.__class__.__name__ == 'VGG':
                    in_ftrs = model.model.classifier.in_features
                    model.model.classifier = nn.Linear(in_ftrs, target_dataset.num_classes)
                    for k, v in checkpoint.items():
                        base_checkpoint[f'classifier.{k.split(".")[-1]}'] = v
                    model.model.load_state_dict(base_checkpoint)
                elif model.model.__class__.__name__ == 'DenseNet' or model.model.__class__.__name__ == 'InceptionV3':
                    in_ftrs = model.model.linear.in_features
                    model.model.linear = nn.Linear(in_ftrs, target_dataset.num_classes)
                    for k, v in checkpoint.items():
                        assert 'bias' in k or 'weight' in k
                        base_checkpoint[f'linear.{k.split(".")[-1]}'] = v
                    model.model.load_state_dict(base_checkpoint)
                else:
                    raise ValueError(f'{model.model.__class__.__name__} not yet supported.')
            else:
                if 'model_state_dict' in checkpoint.keys():
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # models saved with nn.DataParallel have a module. prefix for each key
                    checkpoint = {k[len('module.'):] if 'module.' in k else k:v for k,v in checkpoint.items()}
                    model.model.load_state_dict(checkpoint)
        else:
            # Makes us able to load models saved with legacy versions
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                if 'state_dict' in checkpoint:
                    state_dict_path = 'state_dict'
                else:
                    state_dict_path = 'model_state_dict'

            sd = checkpoint[state_dict_path]
            sd = {k[len('module.'):]:v for k,v in sd.items()}

            model.load_state_dict(sd)
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))

    elif resume_path:
        error_msg = "=> no checkpoint found at '{}'".format(resume_path)
        raise ValueError(error_msg)

    if parallel:
        model = ch.nn.DataParallel(model, device_ids=devices)
    model = model.to(ch.device(f'cuda:{devices[0]}'))

    return model, checkpoint
