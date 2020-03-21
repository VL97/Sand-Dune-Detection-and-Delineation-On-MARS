<table>
  <tr>
    <th>Dune field from NASA's HiRISE </th>
    <th> Final Result from Approach#3 </th> 
      </tr>
  <tr>
    <td>
      <img src="https://github.com/ViditLohia/Sand-Dune-Detection-On-MARS/blob/master/CompressedImages/orig.png" width=500 >
     </td>
    <td>
      <img src="https://github.com/ViditLohia/Sand-Dune-Detection-On-MARS/blob/master/CompressedImages/1%20(1).png" width=500 >
     </td>
    
  </tr>
</table>


This project aims to delineate Martian sand dunes. For more information, refer:
https://github.com/VL97/Sand-Dune-Detection-and-Delineation-On-MARS/wiki

***

3 approaches were mainly used:

APPROACH #1:
We classify slices of images of the dune field as part of/not part of the dune using texture info. Uses K-means classifier.

APPROACH #2:
Same as Approach #1 but uses ANN as a binary classifier.

APPROACH #3:
Uses U-NET segmentation.

***

Approach #1 and #2 are image classification approaches hence limited by:
>>Image traversal is time costly for large resolution images.

>>There is accuracy bottleneck due to tiling effect in such traversal.

>>Database making and Training is time consuming.

Approach #3 is a pixel classifier hence fast and more accurate.

***

Go to respective approach folders and follow 'README' there. 
The project code was tested on Windows 10, python 3.7, keras 2.2.4 and OpenCV 3.4.2.

In case of any problem feel free to contact:

VIDIT LOHIA: 		91+7048906491		f20170632@goa.bits-pilani.ac.in

