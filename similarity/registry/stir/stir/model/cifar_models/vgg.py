'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, num_layers=1):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        
        if num_layers > 1:
            in_ftrs = 512
            fc_layers = []
            for _ in range(num_layers - 1):    fc_layers.append(nn.Linear(in_ftrs, in_ftrs))
            fc_layers.append(nn.Linear(in_ftrs, num_classes))
            self.classifier = nn.Sequential(*fc_layers)
        else:
            self.classifier = nn.Linear(512, num_classes)

    def get_num_layers(self):
        d = list(self.named_parameters())[-1][1].device
        inp = torch.randn((1, 3, 32, 32), device=d)
        return self.forward(inp, return_possible_layers=True)

    def forward(self, x, with_latent=False, fake_relu=False, 
                no_relu=False, layer_num=None, return_possible_layers=False):
        
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"

        output_by_layer = []
        for nb_layer, layer in enumerate(self.features):
            if nb_layer == 0:
                out = layer(x)
            else:
                out = layer(out)
            if isinstance(layer, nn.MaxPool2d) or \
                isinstance(layer, nn.AvgPool2d) or \
                    isinstance(layer, nn.ReLU):
                output_by_layer.append(out)
        out = out.view(out.size(0), -1)
        output_by_layer.append(out)
        
        if with_latent:    latent = out.clone()
        
        out = self.classifier(out)
        output_by_layer.append(out)
        
        if return_possible_layers:
            return len(output_by_layer)

        if layer_num is not None:
            layer_output = output_by_layer[layer_num]
            if len(layer_output.shape) == 4:
                layer_output = torch.mean(layer_output, axis=(1,2))
            elif len(layer_output.shape) == 3:
                layer_output = torch.mean(layer_output, axis=(1))
            return out, layer_output
        elif layer_num is None and with_latent:
            return out, latent
        else:
            return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11(**kwargs):
    return VGG('VGG11', **kwargs)

def VGG13(**kwargs):
    return VGG('VGG13', **kwargs)

def VGG16(**kwargs):
    return VGG('VGG16', **kwargs)

def VGG19(**kwargs):
    return VGG('VGG19', **kwargs)

vgg11 = VGG11
vgg13 = VGG13
vgg16 = VGG16
vgg19 = VGG19
