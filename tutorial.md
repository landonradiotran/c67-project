Detect the Car in Video
=========================


In this activity, you will learn to detect the car in the video.


<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10526061/PCP.gif" width = "480" height = "320">




Follow the given steps to complete this activity:


1. Detect the car.


* Open the main.py file.


* Read the input video file instead of an image


    `video = cv2.VideoCapture("car.mp4")`


* Define infinite while loop.


    `while True:`


* Read the first frame of the video.


    `check, image = video.read()`


* Run the code only if the video frame is read successfully i.e value of check is True.


    `if check:`


* Write condition to detect the car in the image.


    `if labels[classIds[i]] == "car":`


* Change `waitKey` to `1`.


    `cv2.waitKey(1)`


* Quit the display window when the spacebar key is pressed.


    `key = cv2.waitKey(1)`


* Save and run the code to check the output.
