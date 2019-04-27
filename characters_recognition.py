import numpy
import torch
import torch.cuda as cuda
from PIL import Image
from torchvision import transforms


def preproces_iamge(image):
    image = image = Image.fromarray(image.astype('uint8'), 'RGB')
    resize = transforms.Resize((32, 32))
    tensor = transforms.ToTensor()

    image = resize(image)
    image = tensor(image).unsqueeze(0)

    return image


def get_letter_from_index(index):
    category_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z'
                    ]
    return category_map[index]


def predict(image, net):
    image = preproces_iamge(image)
    with torch.no_grad():
        net.eval()
        if cuda.is_available():
            net = net.cuda()
        output = net(image)
        output = numpy.array(output)
        index = numpy.argmax(output)

        characters = get_letter_from_index(index)
        return characters
