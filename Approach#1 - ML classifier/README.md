<table>
  <tr>
    <th>Single Dune Input</th>
    <th>Output</th>
  </tr>
  <tr>
    <td>
     <img src="https://github.com/ViditLohia/Sand-Dune-Detection-On-MARS/blob/master/CompressedImages/ml1.png" height="400" width="400">
    </td>
    <td>
     <img src="https://github.com/ViditLohia/Sand-Dune-Detection-On-MARS/blob/master/CompressedImages/ml2.png" height="400" width="400">
    </td>
  </tr>
</table>


***
'Main K nearest Neigbors.py'  Code trains the classifier on the parameters extracted from lbp( local binary pattern) histograms of the images from 'sand_ripple', 'sand_fine' and 'terrain' folders present in the directory. Sand(ripple or fine) is assigned prediction 1 and terrain 0. The code then takes the input image and traverses it in (100x100) slices and passing lbp parameters of the slices to the classifier and predicting if output is 0 or 1. If its 1 it draws yellow circles on the corners of the taken image slice at its original coordinates in the image_for_output image which is the original input image but read to support rgb drawing.


To get ML based approach output as mentioned in our report:
Follow the steps in 'Main K nearest Neigbors.py' file: 

1.Change 'IMAGE' variable as the input image name.
2.To get more passes change the 'pass_density' variable ( take preferably 1,2,4,6,8...)
3.Run the program and press ENTER when prompted.
4.The LBP parameter can be changed by changing 'radius' and 'n_points'. (Recommended to use 6 and 15 respectively) 

The image is saved by name output.png.
***
The following is the result on the whole dune field:

<table>
  <tr>
    <th><img src="https://github.com/ViditLohia/Sand-Dune-Detection-On-MARS/blob/master/CompressedImages/all.png" width=800>
</th>
  </tr>
</table>




