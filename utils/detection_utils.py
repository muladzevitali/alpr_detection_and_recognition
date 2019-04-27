from yolo.yolo.preprocess import prep_image
from yolo.utils.detector import detect_car, detect_alpr, detect_character


def get_car_coordinates(car_detector, image, config, draw=False, path=True):

    car_detector_height = car_detector.net_info['height']
    car_image_loader = prep_image(image, car_detector_height, path=path)
    car_results = detect_car(car_image_loader, car_detector, config, draw=draw)
    car_results = list(filter(lambda result: result[2] in config.car_labels, car_results))
    if car_results:
        return car_results, car_image_loader[1]
    return None, None


def get_alpr_coordinates(alpr_detector, _car_coordinates, car_image, config, draw=False, first=True):

    alpr_detector_height = alpr_detector.net_info['height']
    alpr_image_loader = prep_image(car_image, alpr_detector_height, path=False)
    alpr_coordinates = detect_alpr(alpr_image_loader, alpr_detector, config, draw=draw)
    if alpr_coordinates:
        if first:
            return alpr_coordinates[0], car_image
        return alpr_coordinates, car_image
    return None, None


def get_character_coordinates(characters_detector, _alpr_coordinates, alpr_image, config, draw=False):
    characters_detector_height = characters_detector.net_info['height']

    alpr_image_loader = prep_image(alpr_image, characters_detector_height, path=False)
    character_coordinates = detect_character(alpr_image_loader, characters_detector, config=config, draw=draw)
    if not character_coordinates == 0:
        return character_coordinates, alpr_image
    return None, None
