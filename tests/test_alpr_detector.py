from utils.configuration import Config
from yolo.utils.loader import load_car_detector, load_alpr_detector
from utils.detection_utils import get_car_coordinates, get_alpr_coordinates
import os
car_detector = load_car_detector(weight_file_path=Config.cars_detector_weight_path,
                                 config_file_path=Config.cars_detector_config_path,
                                 resolution=Config.resolution,
                                 cuda=Config.cuda)

alpr_detector = load_alpr_detector(weight_file_path=Config.alpr_detector_weight_path,
                                   config_file_path=Config.alpr_detector_config_path,
                                   resolution=Config.resolution,
                                   cuda=Config.cuda)


if __name__ == "__main__":
    input_folder_path = "images/"
    total_number_of_alpr = len(os.listdir(input_folder_path))
    recognized_number_of_alpr = 0

    for image_file in os.listdir(input_folder_path):
        image_file_path = os.path.join(input_folder_path, image_file)
        car_coordinates, image = get_car_coordinates(car_detector=car_detector,
                                                     image=image_file_path,
                                                     config=Config,
                                                     path=True)
        if not car_coordinates:
            continue

        for car in car_coordinates:
            car_image = image[car[0][1]: car[1][1], car[0][0]:car[1][0]]
            alpr_coordinates, _ = get_alpr_coordinates(alpr_detector, car, image, config=Config, first=True)
            if alpr_coordinates:
                recognized_number_of_alpr += 1
                break

    print("Found %s of ALPRs out of %s" %(recognized_number_of_alpr, total_number_of_alpr))
