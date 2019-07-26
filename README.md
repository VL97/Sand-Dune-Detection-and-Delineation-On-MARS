<table>
  <tr>
    <th>Dune field from NASA's HiRISE </th>
      </tr>
  <tr>
    <td>
      <img src="https://github.com/ViditLohia/Sand-Dune-Detection-On-MARS/blob/master/CompressedImages/orig.png" width=800 >
     </td>
  </tr>
</table>


This project aims to delineate Martian sand dunes. For more information, refer github-wiki of the project.

***

3 approaches were mainly used:

APPROACH #1:
Uses ML based K-means classifier for image classification.

APPROACH #2:
Uses AI based Artifical Neural Network(ANN) for image classification.

APPROACH #3:
Implemented U-NET segmentation.

***

Approach #1 and #2 are image classification approaches hence limited by:
>>Image traversal is time costly for large resolution images.
>>There is accuracy bottleneck due to tiling effect in such traversal.

We also implemented a Convolutional Neural Network(CNN) for AI based image classification on slices of the main input image but had to 
dump the approach mainly because it suffered additionally from:
>>Database making and Training is time consuming.

Approach #3 is a pixel classifier hence fast and more accurate.

***

Go to respective approach folders and follow 'README' there. 
The project code was tested on Windows 10, python 3.7, keras 2.2.4 and OpenCV 3.4.2.

In case of any problem feel free to contact:

VIDIT LOHIA: 		91+7048906491		f20170632@goa.bits-pilani.ac.in

