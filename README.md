# License plate localization and recognition

The project is split into several parts:
* Localize the car in the image and take the coordinates of the car.
* Localize the license plate in the car image using image and coordinates of the car from previous step.
* Localize the characters in the license plate image from previous step
* Recognize the characters from previous step


For the localizing the cars, we use pre trained model of YOLOv3-608 done on coco dataset. The model contains 80 classes
from which we use 4 classes: Cars, Motorbike, Bus, Truck. Model is the most precise one and can be done only on 20 fps.

The same model is used for localizing the license plate. We trained it on our dataset. Firstly we took car images from an input image and than get the bounding boxes of license plate from car's image and trained the model on that information:
car image + bounding box of license plate. As we wanted the maximum precision the input image was increased to 608 x 608 as it is the maximum input for YOLO. 

For localizing characters we use YOLOv3-tiny. We traind in on oure dataset. We took character bounding boxes from cars license plate image. Input image size is 416 x 416

For recognize the characters we use resnet18 model. We trained it on our dataset. We croped and labeled characters from license plate and trained the model on that data:
Input size is 32x32 and output is character.



The models can be found <a href='https://drive.google.com/open?id=1i4wW_d4oZDp-icTOGNQ2SapdNFnU_2Ky'> here </a>
You should place them into the folder yolo/wights/ folder.

###  Running the program
Install the requirements with 'pip install -r requirements.txt'
Into the file alpr_detection.py change the value INPUT_IMAGE to the image path you want to run detection on.
Then run 'python alpr_detection.py'


scp -i server.pem -r Makefile build darknet src scripts obj darknet.py cfg ubuntu@ec2-3-120-159-143.eu-central-1.compute.amazonaws.com:/home/ubuntu/projects/darknet
scp -i server.pem -r data/characters data/plates.names ubuntu@ec2-3-120-159-143.eu-central-1.compute.amazonaws.com:/home/ubuntu/projects/darknet/data
 