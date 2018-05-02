Main URL of the dataset:
https://figshare.com/collections/Full_Collection_Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3255643

The dataset is split in 5 different zips for download: the images for the 4 different cities and the groundtruth data for all cities.

The folder structure of photovoltaic_array_location_dataset should be:

```
photovoltaic_array_location_dataset
|-- gt
|   |-- polygonDataExceptVertices.csv
|   `-- ...
|-- images
|   |-- fresno
|   |   |-- 11ska325710.tif
|   |   `-- ...
|   |-- modesto
|   |   `-- ...
|   |-- oxnard
|   |   `-- ...
|   |-- stockton
|   |   `-- ...
|   `-- ...
|-- tfrecords.polycnn (created by 0_prepare_dataset.py)
|   |-- train.tfrecords
|   `-- ...
|-- tfrecords.unet_and_vectorization(created by 0_adapt_dataset.py)
|   |-- train.tfrecords
|   `-- ...
`-- README.md (this file)
```

Place the downloaded files to match this structure.