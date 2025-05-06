## Download
Create a folder `IDD_datasets` inside `benchmark/datasets`. 
```bash
mkdir benchmark/datasets/IDD_datasets
```
Due to a licence restriction of source data, we can not provide the final dataset, but it needs to be compiled from the original sources.
Download the following datasets from [IDD website](https://idd.insaan.iiit.ac.in/dataset/download/) and place them in `benchmark/datasets/IDD_datasets` folder:
-    IDD Segmentation (IDD 20k Part I)
-    IDD Segmentation (IDD 20k Part II)
-    IDD-AW
-    IDD-X
```bash
tar xvf ./benchmark/datasets/IDD_datasets/iddaw.tar.gz -C ./benchmark/datasets/IDD_datasets/
tar xvf ./benchmark/datasets/IDD_datasets/idd-20k-II.tar.gz -C ./benchmark/datasets/IDD_datasets/
tar xvf ./benchmark/datasets/IDD_datasets/idd-segmentation.tar.gz -C ./benchmark/datasets/IDD_datasets/
tar xvf ./benchmark/datasets/IDD_datasets/iddx.tar.gz -C ./benchmark/datasets/IDD_datasets/iddx/
```

Then, download ISSU labels `ISSU_datasets.tar.gz` from [here](https://drive.google.com/file/d/1reWmNeRvvY4ohznl6cDPugBL9CsaTjEd/view?usp=sharing) and place it inside `benchmark/datasets`
```bash
gdown 1reWmNeRvvY4ohznl6cDPugBL9CsaTjEd --output ./benchmark/datasets/
tar xvf ./benchmark/datasets/ISSU_datasets.tar.gz -C ./benchmark/datasets/
```

## Compose
To compose the ISSU datasets, run the following command (you can also run for different splits by setting the `--split` to `train/val/test_static/test_temporal`)
```bash
python ./benchmark/datasets/scripts/compose_ISSU.py --IDD_path ./benchmark/datasets/IDD_datasets --ISSU_path ./benchmark/datasets/ISSU_datasets --split all
```

## Convert
We combine both the `ISSU-Test-Static` and `ISSU-Test-Temporal` into a single folder  `dataset_ISSUTrack` with the filename format: `SPLIT_FOLDER_IMAGEROOT.png (e.g. static_0_009599.png)`. This is done by running the following command
```bash
python ./benchmark/datasets/scripts/convert_idd_data.py --ISSU_path ./benchmark/datasets/ISSU_datasets --destination ./benchmark/datasets/dataset_ISSUTrack
```

```
benchmark/
    ├── datasets/
    │   ├── IDD_datasets/
    │   │   ├── IDD_Segmentation
    │   │   ├── idd20kII
    │   │   ├── IDDAW
    │   │   ├── iddx
    |   ├── ISSU_datasets/
    |   |   ├── ISSU_IDD_maps
    |   |   ├── ISSU-Train
    |   |   ├── ISSU-Test-Static
    |   |   ├── ISSU-Test-Temporal
    |   |   ├── ISSU-Test-Temporal-Clip
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
```
