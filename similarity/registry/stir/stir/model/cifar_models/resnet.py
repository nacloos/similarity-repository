'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from stir.model.tools.custom_modules import SequentialWithArgs, FakeReLU
from functools import partial


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x, fake_relu=False, return_output_by_layer=False):
        output_by_layer = []
        out = F.relu(self.bn1(self.conv1(x)))
        output_by_layer.append(out)

        out = self.bn2(self.conv2(out))
        # output_by_layer.append(out)
        out += self.shortcut(x)
        output_by_layer.append(out)

        if return_output_by_layer:
            if fake_relu:
                return FakeReLU.apply(out), output_by_layer
            return F.relu(out), output_by_layer
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, fake_relu=False, return_output_by_layer=False):
        output_by_layer = []
        out = F.relu(self.bn1(self.conv1(x)))
        output_by_layer.append(out)
        out = F.relu(self.bn2(self.conv2(out)))
        output_by_layer.append(out)
        out = self.bn3(self.conv3(out))
        # output_by_layer.append(out)
        out += self.shortcut(x)
        output_by_layer.append(out)

        if return_output_by_layer:
            if fake_relu:
                return FakeReLU.apply(out), output_by_layer
            return F.relu(out), output_by_layer
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)

class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    # num_layers is for additional fully connected layers at the end
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1, num_layers=1, skip_layers=False):
        super(ResNet, self).__init__()

        self.skip_layers = skip_layers

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        if self.skip_layers:
            self._make_skip_layer_shortcut(block, widths, [1, 2, 2, 2])
        if num_layers > 1:
            in_ftrs = feat_scale*widths[3]*block.expansion
            fc_layers = []
            for _ in range(num_layers - 1):    fc_layers.append(nn.Linear(in_ftrs, in_ftrs))
            fc_layers.append(nn.Linear(in_ftrs, num_classes))
            self.linear = nn.Sequential(*fc_layers)
        else:
            self.linear = nn.Linear(feat_scale*widths[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)
    
    def _make_skip_layer_shortcut(self, block, widths, strides):
        ## aligns input to output dims
        in_planes = widths[0]
        for idx, (planes, stride) in enumerate(zip(widths, strides)):
            if stride != 1 or in_planes != block.expansion*planes:
                shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, block.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.expansion*planes)
                )
            else:
                shortcut = nn.Sequential()
            in_planes = planes * block.expansion
            self.__setattr__(f'shortcut_layer{idx+1}', shortcut)

    def get_num_layers(self):
        d = list(self.named_parameters())[-1][1].device
        inp = torch.randn((1, 3, 32, 32), device=d)
        return self.forward(inp, return_possible_layers=True)

    def forward(self, x, with_latent=False, fake_relu=False, 
                no_relu=False, layer_num=None, return_possible_layers=False):
        '''
        return_possible_layers if True, will return number of intermediate layers
        '''
        
        assert (not no_relu),  \
            "no_relu not yet supported for this architecture"
        output_by_layer = []
        out = F.relu(self.bn1(self.conv1(x)))
        output_by_layer.append(out)

        for idx, block in enumerate(self.layer1):
            if idx == 0:
                out_layer, output_by_layer_block = block(out, return_output_by_layer=True)
            else:
                out_layer, output_by_layer_block = block(out_layer, return_output_by_layer=True)
            output_by_layer.extend(output_by_layer_block)
        
        if self.skip_layers:
            out_layer += self.shortcut_layer1(out)
        out = out_layer 

        for idx, block in enumerate(self.layer2):
            if idx == 0:
                out_layer, output_by_layer_block = block(out, return_output_by_layer=True)
            else:
                out_layer, output_by_layer_block = block(out_layer, return_output_by_layer=True)
            output_by_layer.extend(output_by_layer_block)
        
        if self.skip_layers:
            out_layer += self.shortcut_layer2(out)
        out = out_layer

        for idx, block in enumerate(self.layer3):
            if idx == 0:
                out_layer, output_by_layer_block = block(out, return_output_by_layer=True)
            else:
                out_layer, output_by_layer_block = block(out_layer, return_output_by_layer=True)
            output_by_layer.extend(output_by_layer_block)
        
        if self.skip_layers:
            out_layer += self.shortcut_layer3(out)
        out = out_layer

        for idx, block in enumerate(self.layer4):
            if idx == 0:
                out_layer, output_by_layer_block = block(out, fake_relu=fake_relu, return_output_by_layer=True)
            else:
                out_layer, output_by_layer_block = block(out_layer, fake_relu=fake_relu, return_output_by_layer=True)
            output_by_layer.extend(output_by_layer_block)
        
        if self.skip_layers:
            out_layer += self.shortcut_layer4(out)
        out = out_layer

        pre_out = F.avg_pool2d(out, 4)
        pre_out = pre_out.view(pre_out.size(0), -1)
        output_by_layer.append(pre_out)
        # return pre_out
        final = self.linear(pre_out)
        output_by_layer.append(final)

        if return_possible_layers:
            return len(output_by_layer)

        if layer_num is not None:
            layer_output = output_by_layer[layer_num]
            if len(layer_output.shape) == 4:
                layer_output = torch.mean(layer_output, axis=(1,2))
            elif len(layer_output.shape) == 3:
                layer_output = torch.mean(layer_output, axis=(1))
            return final, layer_output
        elif layer_num is None and with_latent:
            return final, pre_out
        else:
            return final


def ResNet18(**kwargs):
    # return torchvision.models.resnet18(pretrained=False, num_classes=10)
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet18Wide(wm, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=wm, **kwargs)

def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wd=.75, **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet34Wide(wm, **kwargs):
    return ResNet(BasicBlock, [3,4,6,3], wm=wm, **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet50Wide(wm, **kwargs):
    return ResNet(Bottleneck, [3,4,6,3], wm=wm, **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)

resnet18 = ResNet18
resnet18skiplayer = partial(ResNet18, **{'skip_layers': True})
resnet34 = ResNet34
resnet34skiplayer = partial(ResNet34, **{'skip_layers': True})
resnet50 = ResNet50
resnet101 = ResNet101
resnet152 = ResNet152
resnet18wide2 = partial(ResNet18Wide, 2)
resnet18wide3 = partial(ResNet18Wide, 3)
resnet18wide4 = partial(ResNet18Wide, 4)
resnet18wide5 = partial(ResNet18Wide, 5)
resnet34wide2 = partial(ResNet34Wide, 2)
resnet34wide3 = partial(ResNet34Wide, 3)
resnet34wide4 = partial(ResNet34Wide, 4)
resnet34wide5 = partial(ResNet34Wide, 5)
resnet50wide2 = partial(ResNet50Wide, 2)
resnet50wide3 = partial(ResNet50Wide, 3)
resnet50wide4 = partial(ResNet50Wide, 4)
resnet50wide5 = partial(ResNet50Wide, 5)

# resnet18thin = ResNet18Thin
def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

