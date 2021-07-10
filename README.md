# image-register
To register a image from UAV to the image from satellite which holds the diffrent histogram,  
we need Histogram specification to pre-process those images before implement SIFT to match the keypoints in images.

Sharpness, Brightness and Edge Extractionare both considered to optimize the result of image register.

Notice that SIFT is available only in opencv-python 3.4.2.16, you are supposed to unistall the newest opencv-python then install the specific version.


