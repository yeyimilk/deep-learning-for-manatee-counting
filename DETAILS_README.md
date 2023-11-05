# Counting Manatee Aggregations using Deep Neural Networks and Anisotropic Gaussian Kernel


# Program Setup step by setp



## Source Images
#### Source video 
[Save the Manatee](https://www.savethemanatee.org/manatees/manatee-webcams/10/16) provides on underwater and above water webcames for watching manatees as well as some video clips. You can download the video we used in this project from [Blue Spring Manatee Webcam Highlights - Above Water (3)](https://www.youtube.com/watch?v=KEIDm1S8qmk&t=2676s) and you can also download the video from [Google Drive](https://drive.google.com/drive/folders/1_VNmEzw0PDOJD07m4ApQ-Zcov_wHcp92?usp=sharing)

#### Generate images 
Once the video has been downloaded, move it into the folder `src/image_generator`, run this script to generate images from the video

`python extract_frames_moviepy.py $video_name$.mp4` (`moviepy` is required to run the script)

A folder named `$video_name$-moviepy` will be generated at the same folder and all the images will be placed in that folder.

#### Images samples generated from the video
<p float="left">
  <img src="./samples/frame0-00-40.00.jpg" width="150" />
  <img src="./samples/frame0-04-00.00.jpg" width="150" /> 
  <img src="./samples/frame0-07-30.00.jpg" width="150" />
</p>

#### Remove duplicated images
* **Extract images features** 

	go into `src/drop_images` folder and run  
  `python extract_features.py $your_image_folder$` 
  
  By running this command, for each of the image in the folder, its features are extracted from the image and saved into a `.pickle` file which has the same name with the image. These `.pickle` files are saved in the new folder of `feature_data` which is a subfolder where your run the command.
  
* **Calculate the distance of the images**

  `python feature_distance_calculations.py ${path_to}$/feature_data`
  
  By running this command, it will calculate the distance among each of the images and the results are saved into a subfolder of the feature data folder, `$path_to$/feature_data/distance_results/distance.pickle`.
  
  
 * **Choose images**
   
    As all the distances among all the images have been calculated, a threshold value can be used to determine weather images should be kept or not.


## Labeling Work
In order to label the images, we employee the tool [Crowd Counting Labeler](https://github.com/Elin24/cclabeler/blob/master/README_en_US.md) and modified part of the code to support line labeling work.

The newly program is also attached in this project in the folder of `/src/cclabeler_line`. For the usage of this tool, please check their [guidance from here](https://github.com/Elin24/cclabeler/blob/master/README_en_US.md)
s