import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo.yolo.util import convert2cpu as cpu
from yolo.yolo.util import predict_transform


def parse_cfg(cfg):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    file = open(cfg, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.strip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].strip()
        else:
            key, value = line.split("=")
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global cuda
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, cuda)
        return prediction


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H * stride, W * stride)
        return x


#

class ReOrgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReOrgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert (H % hs == 0), "The stride " + str(self.stride) + " is not a proper divisor of height " + str(H)
        assert (W % ws == 0), "The stride " + str(self.stride) + " is not a proper divisor of height " + str(W)
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(-2, -3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs, ws)
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(-1, -2).contiguous()
        x = x.view(B, C, ws * hs, H // ws, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, C * ws * hs, H // ws, W // ws)
        return x


def create_modules(blocks):
    net_info = blocks[0]  # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    index = 0  # indexing blocks helps with implementing route  layers (skip connections)
    # Input image channels
    # prev_filters = net_info['channels']
    prev_filters = 3
    output_filters = []
    for block in blocks:
        module = nn.Sequential()

        if block["type"] == "net":
            continue

        # If it's a convolutional layer
        if block["type"] == "convolutional":
            # Get the info about the layer
            activation = block["activation"]
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except Exception:
                batch_normalize = 0
                bias = True

            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the conv layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f"conv_{index}", conv)

            # Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)

            # Check the activation.
            # It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                active = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{index}", active)

        # If it's an up sampling layer
        elif block["type"] == "upsample":
            stride = int(block["stride"])
            # up_sample = nn.Upsample(scale_factor=stride, mode="nearest")
            up_sample = Interpolate(scale_factor=stride, mode="nearest")
            module.add_module(f"upsample_{index}", up_sample)

        # If it is a route layer
        elif block["type"] == "route":
            block["layers"] = block["layers"].split(',')
            # Start  of a route
            start = int(block["layers"][0])
            end = int(block["layers"][1]) if len(block["layers"]) > 1 else 0

            # Positive annotation
            if start > 0:
                start = start - index

            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module(f"route_{index}", route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connection
        elif block["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)

        elif block["type"] == "maxpool":
            stride = int(block["stride"])
            size = int(block["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            module.add_module(F"maxpool_{index}", maxpool)

        # Yolo is the detection layer
        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(each) for each in mask]

            anchors = block["anchors"].split(",")
            anchors = [int(each) for each in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        else:
            print("Something I dunno")
            assert False

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1

    return net_info, module_list


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks

    def get_module_list(self):
        return self.module_list

    def forward(self, x, cuda):
        detections = []
        modules = self.blocks[1:]
        outputs = {}  # We cache the outputs for the route layer

        write = 0
        for i in range(len(modules)):

            module_type = modules[i]["type"]
            types = ["convolutional", "upsample", "maxpool"]
            if module_type in types:
                x = self.module_list[i](x)
                outputs[i] = x

            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x

            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])
                # Get the number of classes
                num_classes = int(modules[i]["classes"])
                # Output the result
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, cuda)
                if type(x) == int:
                    continue
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                outputs[i] = outputs[i - 1]

        try:
            return detections
        except Exception:
            return 0

    def load_weights(self, weightfile):
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        # Open the weights file
        fp = open(weightfile, "rb")
        if 'characters' in weightfile:
            count = 4
        else:
            count = 5
        header = np.fromfile(fp, dtype=np.int32, count=count)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except Exception:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def save_weights(self, savedfile):

        fp = open(savedfile, 'wb')

        # Attach the header at the top of the file
        self.header[3] = self.seen
        header = self.header

        header = header.numpy()
        header.tofile(fp)

        # Now, let us save the weights
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except Exception:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # If the parameters are on GPU, convert them back to CPU
                    # We don't convert the parameter to GPU
                    # Instead. we copy the parameter and then convert it to CPU
                    # This is done as weight are need to be saved during training
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weight.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)

                else:
                    cpu(conv.bias.data).numpy().tofile(fp)

                # Let us save the weights for the Convolutional layers
                cpu(conv.weight.data).numpy().tofile(fp)
