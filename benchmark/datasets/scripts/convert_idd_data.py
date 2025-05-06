import os
import shutil
import argparse
from pathlib import Path
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--ISSU_path', type=str, required=True, help='Path where ISSU dataset is stored')
parser.add_argument('--destination', type=str, required=True, help='Path where the output dataset tracks are stored')
args = parser.parse_args()

dataset_type = ["ISSU-Test-Static", "ISSU-Test-Temporal"]
ground_truth_subpath = "gtFine_csT"
imgs_subpath = "leftImg8bit"
type_dirs = ["test"]

destination_root_path = args.destination
Path(destination_root_path).mkdir(parents=True, exist_ok=True)

for tdir in type_dirs:
    for dtype in dataset_type:
        dataset_prefix = "static" if dtype == "ISSU-Test-Static" else "temporal"
        print(f"\t dataset: {dataset_prefix}")

        out_dir_images = os.path.join(destination_root_path, "images")
        out_dir_labels = os.path.join(destination_root_path, "labels_masks")

        gt_path_t = os.path.join(args.ISSU_path, dtype, ground_truth_subpath, tdir)
        imgs_path_t = os.path.join(args.ISSU_path, dtype, imgs_subpath, tdir)

        sub_dirs = os.listdir(gt_path_t)

        os.makedirs(out_dir_images, exist_ok=True)
        os.makedirs(out_dir_labels, exist_ok=True)

        for subdir in tqdm.tqdm(sub_dirs):
            gt_path = os.path.join(gt_path_t, subdir)
            imgs_path = os.path.join(imgs_path_t, subdir)

            for img_name in os.listdir(imgs_path):
                im_name_short = img_name[:-(len(imgs_subpath)+3+1+1)] # +1"_" + len + 1"." + 3"png"
                im_out_name = dataset_prefix + "_" + subdir + "_" + im_name_short + ".png"
                gt_image_path = os.path.join(gt_path, im_name_short + "_" + ground_truth_subpath + ".png")
                if not os.path.isfile(gt_image_path):
                    continue

                shutil.copyfile(os.path.join(imgs_path, img_name), os.path.join(out_dir_images, im_out_name))
                shutil.copyfile(gt_image_path, os.path.join(out_dir_labels, im_out_name))
