import os

IMAGES = '/home/vitali/Downloads/plates_bbox/Images/001'
LABELS = '/home/vitali/Downloads/plates_bbox/Labels/001'


def delete_empty_labels(folder=LABELS):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        with open(file_path, 'r') as read_file:
            if read_file.readline().strip() == '0':
                print(file)
                os.remove(file_path)

    print('Empty labels deleted')


def delete_unpaired_images(image_folder=IMAGES, labels_folder=LABELS):
    for label in os.listdir(labels_folder):
        delete = True
        image_name = label.split('.')[0]
        for image in os.listdir(image_folder):
            if image_name == image.split('.')[0]:
                delete = False
        if delete:
            os.remove(os.path.join(labels_folder, label))
    print('Unpaired images deleted')


delete_empty_labels()
delete_unpaired_images()
