# image-register
To register a image from UAV to the image from satellite which holds the diffrent histogram,  
we need Histogram specification to preprocess those images before implement SIFT to match the keypoints in images.

Sharpness, Brightness and Edge Extractionare both considered to optimize the result of image register.

Notice that SIFT is available only in **opencv-python 3.4.2.16** , you are supposed to unistall the newest opencv-python then install the specific version.

## The final.py works as below:
### Load images
* Load the real images from UAV and reaference images from satellite into lists.
### Preprocess
* Execute the hist_specification function to make histogram of reals become as close to the ref's as possible
* Optional step: use PIL.ImageEnhance to achieve the goal with higher accuracy.
### Match with Sift
* Figure out the keypoints in two images with sift.detectAndCompute
* Cluster the nearest keypoints with KNN(k-Nearest Neighbors)
### Result ouput
* Caculate dx and dy which are coordinate of the real image in the reference image
* Save the result of image register to the current directory as txt(all dx and dy) and png(real, ref and image registerd for contrast)
