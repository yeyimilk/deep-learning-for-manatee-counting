This folder is just an example and placeholder. 
You have to download the images and labels from [here](https://drive.google.com/drive/folders/1_VNmEzw0PDOJD07m4ApQ-Zcov_wHcp92).

In the dataset, it includes all the images and labels. However, I do not have enough storage in Google drive to upload the density maps. You have to generate the density maps by youself which may takes about 5-10 minutes.

Please run the following script to generate the three types of density map

```
cd src/densitymap_generator
python make_dataset.py
```

The final directory structure should be the same as current one.