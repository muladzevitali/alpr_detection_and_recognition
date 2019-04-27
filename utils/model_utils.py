from yolo.utils.loader import load_car_detector, load_alpr_detector, load_character_detector
from utils.configuration import Config


car_detector = load_car_detector(weight_file_path=Config.cars_detector_weight_path,
                                 config_file_path=Config.cars_detector_config_path,
                                 resolution=Config.resolution,
                                 cuda=Config.cuda)

alpr_detector = load_alpr_detector(weight_file_path=Config.alpr_detector_weight_path,
                                   config_file_path=Config.alpr_detector_config_path,
                                   resolution=Config.resolution,
                                   cuda=Config.cuda)

characters_detector = load_character_detector(weight_file_path=Config.characters_detector_weight_path,
                                              config_file_path=Config.characters_detector_config_path,
                                              resolution=Config.resolution,
                                              cuda=Config.cuda)
