import os

PATH = '/home/vitali/Downloads/logo_batch'
LOGO = "Germany"

for file in os.listdir(PATH):
    if file[0] == '_':
        file_path = os.path.join(PATH, file)
        new_path = os.path.join(PATH, LOGO + file)
        os.rename(file_path, new_path)
