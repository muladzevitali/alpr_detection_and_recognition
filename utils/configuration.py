import torch


class Config:
    cars_detector_weight_path = "yolo/weights/cars.weights"
    cars_detector_config_path = "yolo/configs/cars.cfg"
    cars_detector_labels_path = "yolo/configs/cars.names"
    car_labels = [2, 3, 5, 7]
    alpr_detector_weight_path = "yolo/weights/alpr_old.weights"
    alpr_detector_config_path = "yolo/configs/alpr.cfg"
    alpr_detector_labels_path = "yolo/configs/alpr.names"

    characters_detector_weight_path = "yolo/weights/character_46000.weights"
    characters_detector_config_path = "yolo/configs/characters.cfg"
    characters_detector_labels_path = "yolo/configs/characters.names"

    output_folder = "results"
    cuda = int(torch.cuda.is_available())
    resolution = 416
    scales = "0,1,2"
    batch_size = 1
    confidence = 0.3
    nms_thresh = 0.3

