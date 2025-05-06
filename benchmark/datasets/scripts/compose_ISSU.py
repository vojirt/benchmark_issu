import json
import shutil
from pathlib import Path
import cv2
import argparse
import tqdm


def process_split(args, maps, split):
    Path(f"{args.ISSU_path}/{maps[split]}").mkdir(parents=True, exist_ok=True)

    fp = open(f'{args.ISSU_path}/ISSU_IDD_maps/ISSU_{split}.json')
    ISSU_IDD_map = json.load(fp)
    fp.close()

    if "temporal" not in split:
        for folder in tqdm.tqdm(ISSU_IDD_map.keys()):
            fp_folder = ISSU_IDD_map[folder]["path"]
            imgs = ISSU_IDD_map[folder]["imgs"]
            Path(f"{args.ISSU_path}/{maps[split]}/{folder}").mkdir(parents=True, exist_ok=True)
            for img in imgs:
                img = img.split(".")[0]

                if "IDD_Segmentation" in fp_folder:
                    # images are saved as png
                    shutil.copy(f"{args.IDD_path}/{fp_folder}/{img}.png", f"{args.ISSU_path}/{maps[split]}/{folder}/{img}.png")
                elif "idd20kII" in fp_folder:
                    # images are saved as jpg
                    shutil.copy(f"{args.IDD_path}/{fp_folder}/{img}.jpg", f"{args.ISSU_path}/{maps[split]}/{folder}/{img}.png")
                elif "IDDAW" in fp_folder:
                    # images are saved as imgid_rgb.png
                    img_aw = img.split("_leftImg8bit")[0]
                    shutil.copy(f"{args.IDD_path}/{fp_folder}/{img_aw}_rgb.png", f"{args.ISSU_path}/{maps[split]}/{folder}/{img}.png")

    elif "temporal" in split:
        for folder in tqdm.tqdm(ISSU_IDD_map[split].keys()):
            folder_ISSU = folder.split('/')[0]
            folder_IDDX = folder.split('/')[1]
            img_ids = [int(i.split('_leftImg8bit')[0]) for i in ISSU_IDD_map[split][folder]]
            Path(f"{args.ISSU_path}/{maps[split]}/{folder_ISSU}").mkdir(parents=True, exist_ok=True)

            video = f"{args.IDD_path}/iddx/iddx_videos/{folder_IDDX}.mp4"

            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                print("Error: Could not open video.")
                continue

            for frame_no in img_ids:		
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no-1) # -1 as frame_no were initially saved as 1-index
                _, frame = cap.read()

                frame = frame[:1080,:,:] # take the front view
                img_wp = f"{args.ISSU_path}/{maps[split]}/{folder_ISSU}/{frame_no}_leftImg8bit.png"
                cv2.imwrite(img_wp, frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ISSU_path', type=str, required=True, help='Path where ISSU dataset is stored')
    parser.add_argument('--IDD_path', type=str, required=True, help='Path where IDD datasets are stored')
    parser.add_argument('--split', type=str, required=True, choices=['all','train','val','test_static','test_temporal','test_temporal_clip'],
                        help='Provide the dataset args.split to compose')
    args = parser.parse_args()

    maps = {
            'train': 'ISSU-Train/leftImg8bit/train',
            'val' : 'ISSU-Train/leftImg8bit/val',
            'test_static': 'ISSU-Test-Static/leftImg8bit/test',
            'test_temporal_clip': 'ISSU-Test-Temporal-Clip/leftImg8bit/test',
            'test_temporal': 'ISSU-Test-Temporal/leftImg8bit/test'
            }

    if args.split == "all":
        all_splits = ['train','val','test_static','test_temporal','test_temporal_clip']
        for split in all_splits:
            print(f"Processing split {split} ...")
            process_split(args, maps, split)
    else:
        process_split(args, maps, args.split)


