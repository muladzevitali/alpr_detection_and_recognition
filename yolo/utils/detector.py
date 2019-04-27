from yolo.utils.draw import (rectangle, get_boxes)
from yolo.yolo.util import *


def detect(image_loader, model, labels, cuda, confidence, nms_thresh, draw=False, path=None, output_folder=None):
    classes = load_classes(labels)
    image_processed = image_loader[0].cuda() if cuda else image_loader[0]
    # Original images from batches
    image_original = image_loader[1]
    # Dimension of original image
    height = model.net_info['height']
    # Get predictions
    predictions = predict(model=model,
                          processed_image=image_processed,
                          classes=classes,
                          cuda=cuda,
                          confidence=confidence,
                          nms_thresh=nms_thresh)
    if type(predictions) == int:
        return None
    if cuda:
        torch.cuda.synchronize()
    # Predictions rescaled to the original image
    scaled_predictions = rescale_prediction(image_loader, predictions, height, cuda)
    # Save image in output folder
    if draw:
        rectangle(scaled_predictions, image_original, classes, output_folder, path)
    # Get boxes from detected objects
    boxes = get_boxes(predictions)
    return boxes


def detect_car(image_loader, car_detector, config, draw=False, path=None):

    return detect(image_loader=image_loader, model=car_detector, labels=config.cars_detector_labels_path,
                  cuda=config.cuda, confidence=config.confidence, nms_thresh=config.nms_thresh,
                  draw=draw, path=path, output_folder=config.output_folder)


def detect_alpr(image_loader, alpr_detector, config, draw=False, path=None):

    return detect(image_loader=image_loader, model=alpr_detector,
                  labels=config.alpr_detector_labels_path,
                  cuda=config.cuda, confidence=config.confidence, nms_thresh=config.nms_thresh,
                  draw=draw, path=path, output_folder=config.output_folder)


def detect_character(image_loader, character_detector, config, draw=False, path=None):

    return detect(image_loader=image_loader, model=character_detector, labels=config.characters_detector_labels_path,
                  cuda=config.cuda, confidence=config.confidence, nms_thresh=config.nms_thresh,
                  draw=draw, path=path, output_folder=config.output_folder)


def handle_dimensions(image_loader, predictions, cuda):
    """
    Repeat image dimensions along axis=1, twice and get cuda version
        if cuda is available
    :param predictions:
    :param image_loader: loader of an image
    :param cuda: cuda.is_avaible()
    :return: image dimensions of class torch FloatTensor or cuda version of it
    """
    image_dimensions = image_loader[2]
    image_dimensions = torch.FloatTensor(image_dimensions).repeat(1, 2)
    image_dimensions = image_dimensions.cuda() if cuda else image_dimensions
    image_dimensions = torch.index_select(image_dimensions, 0, predictions[:, 0].long())
    return image_dimensions


def predict(model, processed_image, classes, cuda, confidence, nms_thresh):
    """
    Get predictions from model
    :param model: network model
    :param processed_image: preprocessed image
    :param classes: classes for detection
    :param args: program parameters
    :return: predictions
    """
    num_classes = len(classes)
    with torch.no_grad():
        predictions = model(Variable(processed_image), cuda)
    predictions = write_results(predictions, confidence, num_classes, nms=True, nms_conf=nms_thresh)
    return predictions


def rescale_prediction(image_loader, predictions, height, cuda):
    """
    Rescale predictions to get boxes according to original size of image
    :param image_loader: loader of images
    :param predictions: predictions from network
    :param height: height of input image
    :return: rescaled predictions
    """
    image_dimensions = handle_dimensions(image_loader, predictions, cuda)
    scaling_factor = torch.min(height / image_dimensions, 1)[0].view(-1, 1)
    predictions[:, [1, 3]] -= (height - scaling_factor * image_dimensions[:, 0].view(-1, 1)) / 2
    predictions[:, [2, 4]] -= (height - scaling_factor * image_dimensions[:, 1].view(-1, 1)) / 2

    predictions[:, 1:5] /= scaling_factor

    for i in range(predictions.shape[0]):
        predictions[i, [1, 3]] = torch.clamp(predictions[i, [1, 3]], 0.0, image_dimensions[i, 0])
        predictions[i, [2, 4]] = torch.clamp(predictions[i, [2, 4]], 0.0, image_dimensions[i, 1])

    return predictions
