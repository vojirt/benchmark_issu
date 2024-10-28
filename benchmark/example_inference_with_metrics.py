import os
import numpy as np
from tqdm import tqdm
import cv2 as cv
from evaluation import Evaluation
from types import SimpleNamespace
import torch

def method_dummy(image, **_):
    """ Very naive method: return color saturation """
    image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV_FULL)
    anomaly_p = image_hsv[:, :, 1].astype(np.float32) * (1./255.)
    return anomaly_p

def method_result_loader_PixOOD(method_name, dset_name, fid, results_root_dir):
    # EXAMPLE: 'fid': 'validation_1'
    # load corresponding saved results
    result_file = os.path.join(results_root_dir, method_name, fid + ".npz")
    if not os.path.isfile(result_file):
        raise FileNotFoundError(f"File not found: {result_file}")
    # result file should containt two fields
    results = np.load(result_file)
    return SimpleNamespace(anomaly_scores = results["anomaly_scores"], 
                           pred_id_labels = results["pred_id_labels"]) 



def simple_dbg(method_name, dset_name, fid, results_root_dir):
    # EXAMPLE: 'fid': 'validation_1'
    # load corresponding saved results
    out = fid.split('_')[1:]
    img = out[-1]
    dir = "_".join(out[:-1])
    name = f"{dir}++{img}_gtFine_csT.pth"
    path = f'{results_root_dir}/{name}'
    # print(path, fid)
    d = torch.load(path)
    score = d['ood_score'].numpy()
    preds = d['semseg'].numpy()
    # results = np.load(result_file)
    return SimpleNamespace(anomaly_scores = score,
                           pred_id_labels = preds)

def main():
    method_name = "EAM"
    # dataset_name = 'IDDObstacleTrack-static'
    # dataset_name = 'IDDAnomalyTrack-static'
    # dataset_name = 'IDDAnomalyTrack-all'
    dataset_name = 'IDDAnomalyFullTrack-static'

    ev = Evaluation(
        method_name = method_name, 
        dataset_name = dataset_name,
    )

    # for frame in tqdm(ev.get_frames()):
    #     # run method here
    #     # result = method_dummy(frame.image)
    #     # result = method_result_loader_PixOOD(method_name, dataset_name, frame.fid, results_root_dir="./_results")
    #     result = simple_dbg(method_name, dataset_name, frame.fid, results_root_dir="/mnt/sdb1/mgrcic/experiments/dbg_eam_results")
    #     # provide the output for saving
    #     ev.save_output(frame, result)
    #
    # # wait for the background threads which are saving
    # ev.wait_to_finish_saving()

    print("Calculating pixel-level open-set metrics")
    ev.calculate_metric_from_saved_outputs(
        'IntersectionOverUnion',
        frame_vis=False,
        parallel=True,
        load_closed_set_preds=True,
        threshold=0.5,
    )

    print("Calculating pixel-level metrics")
    ev.calculate_metric_from_saved_outputs(
        'PixBinaryClass',
        frame_vis=True,
    )

    print("Calculating instance-level metrics")
    ev.calculate_metric_from_saved_outputs(
        'SegEval-ObstacleTrack',
        frame_vis=True,
    )

if __name__ == '__main__':
    main()
