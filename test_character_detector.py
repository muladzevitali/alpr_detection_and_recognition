import os

from utils.configuration import Config
from utils.detection_utils import get_car_coordinates, get_alpr_coordinates, get_character_coordinates
from utils.model_utils import car_detector, alpr_detector, characters_detector

input_folder_path = "images/test_data"
# Filter only image files in input folder
image_files = [file for file in os.listdir(input_folder_path) if file.split('.')[-1] in ["jpg", "png", 'jpeg']]
# Get total number of images
total_number_of_characters = len(image_files) * 7
# Set counter on detected alprs
recognized_number_of_characters = 0

print("\n", "-" * 17, 'Starting Testing Process', "-" * 17)

for _index, _image_path in enumerate(image_files):
    try:
        image_path = os.path.join(input_folder_path, _image_path)
        # Get car coordinates for input image
        car_coordinates, image = get_car_coordinates(car_detector=car_detector, image=image_path, config=Config)

        # If no cars where detected continue
        if not car_coordinates:
            continue
        # Get alpr for each found car

        print('(%s / %s) Car detected on image %s' % (_index + 1, len(image_files), _image_path))
        for car in car_coordinates:
            # Get the car image from input image
            car_image = image[car[0][1]: car[1][1], car[0][0]:car[1][0]]
            # Get alpr coordinates in car image
            alpr_coordinates, _ = get_alpr_coordinates(alpr_detector, car, image, config=Config, first=True)
            if not alpr_coordinates:
                continue

            print('(%s / %s) ALPR detected on image %s' % (_index + 1, len(image_files), _image_path))
            # Get image of alpr from car image
            alpr_image = image[alpr_coordinates[0][1]: alpr_coordinates[1][1],
                         alpr_coordinates[0][0]:alpr_coordinates[1][0]]
            # Get characters coordinates in alpr image

            characters_coordinates, _ = get_character_coordinates(characters_detector, alpr_coordinates, alpr_image,
                                                                  config=Config)
            # Increase the counter with found number of objects
            recognized_number_of_characters += len(characters_coordinates)
            # Continue to next image if characters are recognized
            break
    except:
        total_number_of_characters -= 7
        continue

print("Found %s of Characters out of %s, (%s Percent)" % (total_number_of_characters, recognized_number_of_characters,
                                                          recognized_number_of_characters / total_number_of_characters * 100))
