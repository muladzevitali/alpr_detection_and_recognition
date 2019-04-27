from yolo.yolo.darknet import Darknet
from yolo.yolo.preprocess import prep_image
import os


def load_model(weight_file_path, config_file_path, resolution, cuda):

    # Set up the neural network
    model = Darknet(config_file_path)
    model.load_weights(weight_file_path)

    # Set height for model input
    model.net_info["height"] = int(resolution)
    assert model.net_info["height"] % 32 == 0
    assert model.net_info["height"] > 32

    # If there's a GPU available, put the model on GPU
    if cuda:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()
    return model


def load_car_detector(weight_file_path, config_file_path, resolution, cuda):
    return load_model(weight_file_path, config_file_path, resolution, cuda)


def load_alpr_detector(weight_file_path, config_file_path, resolution, cuda):
    return load_model(weight_file_path, config_file_path, resolution, cuda)


def load_character_detector(weight_file_path, config_file_path, resolution, cuda):
    return load_model(weight_file_path, config_file_path, resolution, cuda)


def load_images(args, path=None):
    """
    Load images and create output folder
    :param path:
    :param args: program parameters
    :return: image
    """
    # Detection phase
    if path:
        image = path
    else:
        image = os.path.join(os.getcwd(), args.image)
    image_resolution = int(args.resolution)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    return prep_image(image, image_resolution)
