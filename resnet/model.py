from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import (Bottleneck, BasicBlock, model_urls)

_NUM_CLASSES_KEY = 'num_classes'


class ResNetModule(nn.Module):
    """ResNet module for siamese and triplet training and embedding vector"""

    def __init__(self, block, layers, in_channels=3, num_classes=1000):
        self.inplanes = 64
        super(ResNetModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, input_tensor):
        """Extracts features from input tensor
          Args:
            input_tensor - input image tensor
          Returns:
            features_tensor - features tensor
        """
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features_tensor = self.layer4(x)

        return features_tensor

    def embedding(self, input_tensor):
        """Extracts features vector for passed tensor
          Args:
            input_tensor - input image tensor
          Returns:
            output_tensor - extracted features vector
        """
        x = self.features(input_tensor)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output_tensor = self.fc(x)

        return output_tensor

    def double_call(self, x1, x2):
        """Call for siamese result
            Args:
                x1 - first input
                x2 - second input
            Returns:
                tuple of -
                    output1 - result for first input
                    output2 - result for second input
        """
        output1 = self.embedding(x1)
        output2 = self.embedding(x2)

        return output1, output2

    def triple_call(self, x1, x2, x3):
        """Call for triplet result
            Args:
                x1 - first input
                x2 - second input
                x3 - third input
            Returns:
                tuple of -
                    output1 - result for first input
                    output2 - result for second input
                    output3 - result for third input
        """
        output1, output2 = self.double_call(x1, x2)
        output3 = self.embedding(x3)

        return output1, output2, output3

    def forward(self, *input_tensor):
        if len(input_tensor) == 1:
            output_tensor = self.embedding(*input_tensor)
        elif len(input_tensor) == 2:
            output_tensor = self.double_call(*input_tensor)
        elif len(input_tensor) == 3:
            output_tensor = self.triple_call(*input_tensor)
        else:
            output_tensor = super(ResNetModule, self).forward(*input_tensor)

        return output_tensor


def _init_layers(layers):
    """Sets default values to layers
        Args:
            layers - layers for ResNet module
        Returns:
            default value if layers are not defined
    """
    return [2, 2, 2, 2] if layers is None else layers


def _add_default_num_classes(named_args, pretrained):
    """Set number of output classes
        Args:
            kwargs - named arguments with output classes
        Returns:
            num_classes - custom number of output classes
    """
    num_classes = named_args.get(_NUM_CLASSES_KEY, 36)
    if pretrained and num_classes != 1000:
        named_args[_NUM_CLASSES_KEY] = 1000

    return num_classes


def _add_custom_num_classes(model, num_classes):
    """Adds custom output classes to model
        Args:
            model - network model
            num_classes - number of output classes
    """
    if num_classes != 1000:
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in, num_classes)


def _init_model(core_type=ResNetModule, block=BasicBlock, layers=None,
                model_key='resnet18', pretrained=False, **kwargs):
    """Initializes appropriated model
      Args:
        core_type - type for model core initialization
        block - block for layers initialization
        layers - model layers
        model_key - key for model URL dictionary
        pretrained - flag for trained weights
        kwargs - additional arguments
      Returns:
        model - network model with weights
    """
    num_classes = _add_default_num_classes(kwargs, pretrained)
    model = core_type(block, _init_layers(layers), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_key]))
        _add_custom_num_classes(model, num_classes)

    return model


def _init_module(block=BasicBlock, layers=None, model_key='resnet18',
                 pretrained=False, **kwargs):
    """Initializes appropriated model
      Args:
        block - block for layers initialization
        layers - model layers
        pretrained - flags for trained weights
        kwargs - additional arguments
      Returns:
        network model with weights
    """
    return _init_model(core_type=ResNetModule, block=block, layers=layers,
                       model_key=model_key, pretrained=pretrained, **kwargs)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
      Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
      Returns:
        network model width weights
    """
    return _init_module(block=BasicBlock, layers=[2, 2, 2, 2], model_key=resnet18.__name__,
                        pretrained=pretrained, **kwargs)
