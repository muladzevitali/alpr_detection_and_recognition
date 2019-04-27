import os

import cv2

from utils.configuration import Config
from utils.detection_utils import get_car_coordinates, get_alpr_coordinates
from yolo.utils.loader import load_car_detector, load_alpr_detector

car_detector = load_car_detector(weight_file_path=Config.cars_detector_weight_path,
                                 config_file_path=Config.cars_detector_config_path,
                                 resolution=Config.resolution,
                                 cuda=Config.cuda)

alpr_detector = load_alpr_detector(weight_file_path=Config.alpr_detector_weight_path,
                                   config_file_path=Config.alpr_detector_config_path,
                                   resolution=Config.resolution,
                                   cuda=Config.cuda)

if __name__ == "__main__":
    # Folder for testing the ALPR detector
    input_folder_path = "images/test_data"
    # Filter only image files in input folder
    image_files = [file for file in os.listdir(input_folder_path) if file.split('.')[-1] in ["jpg", "png", 'jpeg']]
    # Get total number of images
    total_number_of_alpr = len(image_files)
    # Set counter on detected alprs
    recognized_number_of_alpr = 0

    print("\n", "-" * 17, 'Starting Testing Process', "-" * 17)

    for _index_of_image, image_file in enumerate(image_files):
        try:
            # Get exact path to image
            image_file_path = os.path.join(input_folder_path, image_file)
            image = cv2.imread(image_file_path)

            # Get car coordinates and image from input image path and car detector
            car_coordinates, _ = get_car_coordinates(car_detector=car_detector,
                                                     image=image_file_path,
                                                     config=Config,
                                                     path=True)
            # If no car found on the image continue to next image
            if not car_coordinates:
                continue
            print('(%s / %s) Car detected on image %s' % (_index_of_image, total_number_of_alpr, image_file))
            print("-" * 17)
            # Loop over all cars in image
            for _index_of_car, car in enumerate(car_coordinates):
                # Crop car from image
                car_image = image[car[0][1]: car[1][1], car[0][0]:car[1][0]]
                # Get alpr coordinates from car image
                alpr_coordinates, _ = get_alpr_coordinates(alpr_detector, car, image, config=Config, first=True)
                # If alpr was found on the car, increase the alpr counter and continue to next image
                if alpr_coordinates:
                    print('> ALPR detected on image %s' % image_file)
                    print("-" * 17)
                    recognized_number_of_alpr += 1
                    break
        except:
            total_number_of_alpr -= 1
            continue

    print("Found %s of ALPRs out of %s" % (recognized_number_of_alpr, total_number_of_alpr))
