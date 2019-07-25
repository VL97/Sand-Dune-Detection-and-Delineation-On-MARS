'Main K nearest Neigbors.py' CODE TRAINS THE CLASSIFIER ON THE PARAMETERS EXTRACTED FROM LBP( Local Binary Pattern)
HISTOGRAMS OF THE IMAGES FROM 'SAND_RIPPLE', 'SAND_FINE' AND 'TERRAIN' FOLDERS PRESENT IN THE 
DIRECTORY. SAND(RIPPLE OR FINE) IS ASSIGNED PREDICTION 1 AND TERRAIN 0.
THE CODE THEN TAKES THE INPUT IMAGE AND TRAVERSES IT IN (100X100) SLICES 
AND PASSING LBP PARAMETERS OF THE SLICES TO THE CLASSIFIER AND PREDICTING IF 
OUTPUT IS 0 OR 1. IF ITS 1 IT DRAWS YELLOW CIRCLES ON THE CORNERS OF THE TAKEN 
IMAGE SLICE AT ITS ORIGINAL COORDINATES IN THE IMAGE_FOR_OUTPUT IMAGE WHICH IS 
THE ORIGINAL INPUT IMAGE BUT READ TO SUPPORT RGB DRAWING.


To get ML based approach output as mentioned in our report:
Follow the steps in 'Main K nearest Neigbors.py' file: 

1.Change 'IMAGE' variable as the input image name.
2.To get more passes change the 'pass_density' variable ( take preferably 1,2,4,6,8...)
3.Run the program and press ENTER when prompted.
4.The LBP parameter can be changed by changing 'radius' and 'n_points'. (Recommended to use 6 and 15 respectively) 

The image is saved by name output.png.


