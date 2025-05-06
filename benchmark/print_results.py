import os
import argparse
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from pandas import DataFrame, Series

from datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file

experiments_info = {
        "IntersectionOverUnion" : {
            "file_fmt" : "OpenSet_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["ISSUAnomalyTrack"],
        },
        "PixBinaryClass": {
            "file_fmt" : "PixClassCurve_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["ISSUObstacleTrack", "ISSUAnomalyTrack"],
        },
        "SegEval": {
            "file_fmt" : "SegEvalResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["ISSUObstacleTrack", "ISSUAnomalyTrack"],
        },
        "SegEval-TooSmall": {
            "file_fmt" : "SegEval-TooSmallResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["ISSUObstacleTrack", "ISSUAnomalyTrack"],
        },
        "SegEval-Small": {
            "file_fmt" : "SegEval-SmallResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["ISSUObstacleTrack", "ISSUAnomalyTrack"],
        },
        "SegEval-Large": {
            "file_fmt" : "SegEval-LargeResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["ISSUObstacleTrack", "ISSUAnomalyTrack"],
        },
        "SegEval-VeryLarge": {
            "file_fmt" : "SegEval-VeryLargeResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["ISSUObstacleTrack", "ISSUAnomalyTrack"],
        },
}

def get_results_for_exp(exp, method_name, dataset_name, result_dir="./outputs/"):
    filename = os.path.join(result_dir, exp, "data", experiments_info[exp]["file_fmt"].format(method_name=method_name, dataset_name=dataset_name))
    res_dict = {}
    if os.path.isfile(filename):
        tmp_data = hdf5_read_hierarchy_from_file(filename)
        if exp == "PixBinaryClass":
            res_dict["AP"] = 100*tmp_data.area_PRC 
            res_dict["FPR@95TPR"] = 100*tmp_data.tpr95_fpr 
            res_dict["TPR@5FPR"] = 100*tmp_data.fpr5_tpr 
        elif exp == "SegEval":
            res_dict["mF1(all)"] = 100*tmp_data.f1_mean
            res_dict["sIoU_gt(all)"] = 100*tmp_data.sIoU_gt
            res_dict["PPV(all)"] = 100*tmp_data.prec_pred
        elif exp == "SegEval-TooSmall":
            res_dict["mF1(TooSmall)"] = 100*tmp_data.f1_mean
            res_dict["sIoU_gt(TooSmall)"] = 100*tmp_data.sIoU_gt
            res_dict["PPV(TooSmall)"] = 100*tmp_data.prec_pred
        elif exp == "SegEval-Small":
            res_dict["mF1(Small)"] = 100*tmp_data.f1_mean
            res_dict["sIoU_gt(Small)"] = 100*tmp_data.sIoU_gt
            res_dict["PPV(Small)"] = 100*tmp_data.prec_pred
        elif exp == "SegEval-Large":
            res_dict["mF1(Large)"] = 100*tmp_data.f1_mean
            res_dict["sIoU_gt(Large)"] = 100*tmp_data.sIoU_gt
            res_dict["PPV(Large)"] = 100*tmp_data.prec_pred
        elif exp == "SegEval-VeryLarge":
            res_dict["mF1(VeryLarge)"] = 100*tmp_data.f1_mean
            res_dict["sIoU_gt(VeryLarge)"] = 100*tmp_data.sIoU_gt
            res_dict["PPV(VeryLarge)"] = 100*tmp_data.prec_pred
        elif exp == "IntersectionOverUnion":
            res_dict["closed-mIoU"] = 100*tmp_data.closed_miou 
            res_dict["open-mIoU@best_f1"] = 100*tmp_data.open_miou["best_f1"]
            res_dict["open-mIoU@tpr95"] = 100*tmp_data.open_miou["tpr95"]
            res_dict["open-mIoU@fpr05"] = 100*tmp_data.open_miou["fpr05"]
    return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='Rba_pt_ood,Rba_ft_idd_ood_latest', required=False, help='comma seperated method names; should be same as used in example_inference_with_metrics script')
    parser.add_argument('--result_path', type=str, required=False, default='./outputs/', help='path to stored benchmarked results')
    parser.add_argument('--protocol', type=str, default='all', choices=['obstacle', 'anomaly', 'all'], required=True)
    parser.add_argument('--lighting_var', action='store_true', help='evaluate results per lighting variations')
    parser.add_argument('--anom_size_var', action='store_true', help='evaluate results per anomaly size variations')
    args = parser.parse_args()

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

    experiments = ["PixBinaryClass", "IntersectionOverUnion", "SegEval"] 
    if args.anom_size_var:
        experiments += ["SegEval-TooSmall", "SegEval-Small", "SegEval-Large", "SegEval-VeryLarge"]

    tables = {} 
    for dataset_name in dataset_names:
        data = []
        for method_name in method_names:
            res_dict = {}
            for exp in experiments:
                if dataset_name.split('-')[0] not in experiments_info[exp]["valid_datasets"]:
                    continue

                res_dict.update(get_results_for_exp(exp, method_name, dataset_name, result_dir=args.result_path))

            data.append(Series(res_dict, name=method_name))

        tables[dataset_name] = DataFrame(data=data)

    with open(os.path.join(args.result_path, "results.md"), 'w') as f:
        print("\033[104m" + f"Results saved to {os.path.join(args.result_path, 'results.md')}" + "\033[0m")
        for dataset_name in dataset_names:
            print(f"===== {dataset_name} =====")
            print(tables[dataset_name].to_markdown(floatfmt=".2f"))
            print("\n")
            f.write(f"===== {dataset_name} =====\n")
            f.write(tables[dataset_name].to_markdown(floatfmt=".2f"))
            f.write("\n")
            f.write("\n")
