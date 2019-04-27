from detection.utils.flags import init_detection_config
from yolo.utils.detector import detection
from yolo.utils.loader import load_model
from yolo.yolo.preprocess import prep_image
import cv2
import os
_OK = 'ok'


def single_image(_image_path):
    image_loader = prep_image(_image_path, height, path=True)
    results = detection(image_loader, car_detector, flags, draw=False)
    if not results == 0:
        return list(filter(lambda result: result[2] in CAR_LABELS, results))
    return None


def get_coordinates(test_file_path):
    with open(test_file_path, 'r') as _input_file:
        _rows = _input_file.readlines()[1:]
    _coordinates = []

    for row in _rows:
        _row_coordinates = [int(each.strip()) for each in row.split(' ')]
        _coordinates.append(_row_coordinates)
    return _coordinates


def check_if_inside(_car_coordinates, _plate_coordinates):
    if _car_coordinates[0][0] > _plate_coordinates[0]:
        return False
    elif _car_coordinates[0][1] > _plate_coordinates[1]:
        return False
    elif _car_coordinates[1][0] < _plate_coordinates[2]:
        return False
    elif _car_coordinates[1][1] < _plate_coordinates[3]:
        return False
    else:
        return True


if __name__ == '__main__':
    CAR_LABELS = [2, 3, 5, 7]
    # Allowed extensions list
    flags = init_detection_config()
    # Detector object
    car_detector = load_model(flags)
    height = car_detector.net_info['height']
    # Start serving
    # single_image('images/cars-1.jpg')
    IMAGES_PATH = '/home/vitali/Downloads/plates_bbox/Images_c/001'
    LABELS_PATH = '/home/vitali/Downloads/plates_bbox/Labels_c/001'
    images = os.listdir(IMAGES_PATH)
    for _index, image in enumerate(images):
        print(f'{_index} / {len(images)}')
        image_path = os.path.join(IMAGES_PATH, image)
        frame = cv2.imread(image_path)
        cars_coordinates = single_image(image_path)
        text_file_path = image_path.replace('Images_c', 'Labels_c').replace('png', 'txt')
        text_file_name = text_file_path.split('/')[-1]
        plate_numbers_coordinates = get_coordinates(text_file_path)

        for _car_coordinates in cars_coordinates:
            number_of_plates = 0
            coordinates = []
            cropped_image = None
            new_coordinates = []
            for plate_number_coordinates in plate_numbers_coordinates:
                if check_if_inside(_car_coordinates, plate_number_coordinates):
                    print(_car_coordinates, plate_number_coordinates, '**************************')
                    number_of_plates += 1
                    top_left, bottom_right = _car_coordinates[0:2]
                    cropped_image = frame[top_left[1]:bottom_right[1], top_left[0]: bottom_right[0]]
                    new_coordinates = [plate_number_coordinates[0] - _car_coordinates[0][0],
                                       plate_number_coordinates[1] - _car_coordinates[0][1],
                                       -_car_coordinates[0][0] + plate_number_coordinates[2],
                                       -_car_coordinates[0][1] + plate_number_coordinates[3]]
                    coordinates.append(new_coordinates)
            if number_of_plates > 0:
                _image_new_path = os.path.join('/home/vitali/Downloads/plates_bbox/Images/001', image)
                print(_image_new_path)
                cv2.imwrite(_image_new_path, cropped_image)
                _text_file_new_path = os.path.join('/home/vitali/Downloads/plates_bbox/Labels/001', text_file_name)
                with open(_text_file_new_path, 'w') as _output_file:
                    _output_file.write(str(number_of_plates) + '\n')
                    for each in coordinates:
                        print(each)
                        _string = f'{each[0]} {each[1]} {each[2]} {each[3]}'
                        _output_file.write(_string + '\n')
