import os
import numpy as np
import argparse
from tqdm import tqdm
import cv2 as cv
from evaluation import Evaluation
from types import SimpleNamespace
import torch
from joblib import Parallel, delayed 
from pandas import DataFrame, Series

from datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file
from print_results import get_results_for_exp


def result_loader(method_name, dset_name, fid, results_root_dir):
    # load corresponding saved results
    result_file = os.path.join(results_root_dir, method_name, fid + ".npz")
    if not os.path.isfile(result_file):
        raise FileNotFoundError(f"File not found: {result_file}")
    # result file should containt two fields
    results = np.load(result_file)
    return SimpleNamespace(anomaly_scores = results["anomaly_scores"], 
                           pred_id_labels = results["pred_id_labels"]) 

def main(method_name, dataset_name, exps, load_fn, results_root_dir, recompute_results, num_workers):
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

    res_dict = {}
    # exps = ["PixBinaryClass", "SegEval"] #, "SegEval-TooSmall", "SegEval-Small", "SegEval-Large", "SegEval-VeryLarge"]
    for exp in exps:
        print(f"Calculating {exp} metrics")
        ev.calculate_metric_from_saved_outputs(
            exp,
            frame_vis=False,
        )
        # for table plot
        res_dict.update(get_results_for_exp(exp, method_name, dataset_name))

    if "Anomaly" in dataset_name:
        exp = "IntersectionOverUnion"
        print(f"Calculating {exp} metrics")
        ev.calculate_metric_from_saved_outputs(
            exp,
            frame_vis=False,
            parallel=True,
            load_closed_set_preds=True,
        )
        res_dict.update(get_results_for_exp(exp, method_name, dataset_name))

    table = DataFrame(data=[Series(res_dict, name=method_name)])
    print(f"\n===== {dataset_name} =====")
    print(table.to_markdown(floatfmt=".2f"))
    print("\n")


if __name__ == '__main__':
    num_workers = 16
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='Rba_pt_ood,Rba_ft_idd_ood_latest', required=False)
    parser.add_argument('--result_path', type=str, required=True, help='path to stored predictions')
    parser.add_argument('--protocol', type=str, default='all', choices=['obstacle', 'anomaly', 'all'], required=True)
    parser.add_argument('--lighting_var', action='store_true', help='evaluate results per lighting variations')
    parser.add_argument('--anom_size_var', action='store_true', help='evaluate results per anomaly size variations')
    parser.add_argument('--recompute_results', action='store_false', help='recompute results')
    args = parser.parse_args()

    # method_names = [args.method]
    method_names = args.methods.split(',')

    if args.protocol == 'obstacle':
        dataset_names = ["ISSUObstacleTrack-static", "ISSUObstacleTrack-temporal"]
    elif args.protocol == 'anomaly':
        dataset_names = ["ISSUAnomalyTrack-static", "ISSUAnomalyTrack-temporal"]
    elif args.protocol == 'all':
        dataset_names = ["ISSUObstacleTrack-static", "ISSUObstacleTrack-temporal"]
        dataset_names += ["ISSUAnomalyTrack-static", "ISSUAnomalyTrack-temporal"]

    if args.lighting_var:
        dataset_lighting = []
        for dataset in dataset_names:
            dataset_lighting.append(f"{dataset}Normal")
            dataset_lighting.append(f"{dataset}LowLight")

        dataset_names += dataset_lighting

    exps = ["PixBinaryClass", "SegEval"] 
    if args.anom_size_var:
        exps += ["SegEval-TooSmall", "SegEval-Small", "SegEval-Large", "SegEval-VeryLarge"]

    for method in method_names:
        for dataset in dataset_names:
            print("\033[104m" + f"Evaluating method {method} on dataset split {dataset}" + "\033[0m")
            main(method, dataset, exps, result_loader, args.result_path, args.recompute_results, num_workers)
