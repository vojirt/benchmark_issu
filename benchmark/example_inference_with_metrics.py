import os
import numpy as np
from tqdm import tqdm
import cv2 as cv
from evaluation import Evaluation
from types import SimpleNamespace
import torch
from joblib import Parallel, delayed 


def method_dummy(image, **_):
    """ Very naive method: return color saturation """
    image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV_FULL)
    anomaly_p = image_hsv[:, :, 1].astype(np.float32) * (1./255.)
    return anomaly_p

def result_loader_PixOOD(method_name, dset_name, fid, results_root_dir):
    # load corresponding saved results
    result_file = os.path.join(results_root_dir, method_name, fid + ".npz")
    if not os.path.isfile(result_file):
        raise FileNotFoundError(f"File not found: {result_file}")
    # result file should containt two fields
    results = np.load(result_file)
    return SimpleNamespace(anomaly_scores = results["anomaly_scores"], 
                           pred_id_labels = results["pred_id_labels"]) 

def result_loader_EAM(method_name, dset_name, fid, results_root_dir):
    out = fid.split('_')[1:]
    img = out[-1]
    dir = "_".join(out[:-1])
    name = f"{dir}++{img}_gtFine_csT.pth"
    path = f'{results_root_dir}/{name}'
    d = torch.load(path)
    score = d['ood_score'].numpy()
    preds = d['semseg'].numpy()
    return SimpleNamespace(anomaly_scores = score,
                           pred_id_labels = preds)


def main(method_name, dataset_name, load_fn, results_root_dir, recompute_results, num_workers):
    ev = Evaluation(
        method_name = method_name, 
        dataset_name = dataset_name,
        num_workers=num_workers,
    )
    
    if recompute_results:
        print("Computing results ...")
        if "result_loader" in load_fn.__name__:
            print("Detected results loader function, processing will be parallel")
            def process_frame(frame):
                result = load_fn(method_name, dataset_name, frame.fid, results_root_dir)
                ev.save_output(frame, result)
            Parallel(n_jobs=num_workers)(delayed(process_frame)(frame) for frame in tqdm(ev.get_frames())) 
        else:
            for frame in tqdm(ev.get_frames()):
                result = load_fn(method_name, dataset_name, frame.fid, results_root_dir)
                # provide the output for saving
                ev.save_output(frame, result)
    else:
        print("Skipping results computation.")

    if "Full" in dataset_name:
        ev.calculate_metric_from_saved_outputs(
            'IntersectionOverUnion',
            frame_vis=False,
            parallel=True,
            load_closed_set_preds=True,
        )
    else:
        print("Calculating pixel-level metrics")
        ev.calculate_metric_from_saved_outputs(
            'PixBinaryClass',
            frame_vis=False,
        )
        print("Calculating instance-level metrics")
        ev.calculate_metric_from_saved_outputs(
            'SegEval-ObstacleTrack',
            frame_vis=False,
        )

if __name__ == '__main__':
    method_names = ["EAM"]
    load_functions = {'EAM': result_loader_EAM}
    # NOTE: Regular track (without 'Full' in name) needts to be computed first to estimate threholds for the 'Full' evals open-miou metric
    dataset_names = ['IDDAnomalyTrack-static', 'IDDAnomalyFullTrack-static']
    recompute_results = True
    for method in method_names:
        for dataset in dataset_names:
            print("\033[104m" + f"Evaluating method {method} on dataset split {dataset}" + "\033[0m")
            main(method, dataset, load_functions[method], recompute_results)

