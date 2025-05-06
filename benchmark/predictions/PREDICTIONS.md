## Predictions
For each image in `datasets/dataset_ISSUTrack`, compute the following variables:
-    `anomaly_scores`: a 2D matrix containing anomaly score prediction per pixel. 
-    `pred_id_labels`: a 2D matrix containing known label (0-19) prediction per pixel.

Save the predictions in a `.npz` file with the corresponding variable names as `kwds`. The filename follows the same format as the input image filename `e.g. static_0_009599.png -> static_0_009599.npz `. Predictions for an anomaly segmentation method, `YOUR_METHOD` are saved in corresponding folder in `benchmark/predictions/YOUR_METHOD`. An example folder containing predictions of [EAM](https://github.com/matejgrcic/Open-set-M2F) can be downloaded from [here](https://drive.google.com/file/d/1k3n-k1cnePis1imulOnGKB6i1etyJ2xQ/view?usp=sharing).

```
benchmark/
    ├── datasets/
    |   └── dataset_ISSUTrack/
    |   |   ├── labels_masks
    |   |   ├── images
    |   |   |   ├── static_0_009599.png
    |   |   |   ├── static_0_020764.png
    |   |   |   ├── .
    |   |   |   ├── .
    |   |   |   ├── .
    |   |   |   ├── temporal_99_520.png
    |   |   |   ├── temporal_99_550.png
    └── predictions/
    |   ├── EAM
    |   ├── YOUR_METHOD
    |   |   ├── static_0_009599.npz
    |   |   ├── static_0_020764.npz
    |   |   ├── .
    |   |   ├── .
    |   |   ├── .
    |   |   ├── temporal_99_520.npz
    |   |   ├── temporal_99_550.npz
```
