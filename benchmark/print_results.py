import os
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from pandas import DataFrame, Series

from datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file

experiments_info = {
        "IntersectionOverUnion" : {
            "file_fmt" : "OpenSet_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["IDDAnomalyTrack"],
        },
        "PixBinaryClass": {
            "file_fmt" : "PixClassCurve_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["IDDObstacleTrack", "IDDAnomalyTrack"],
        },
        "SegEval": {
            "file_fmt" : "SegEvalResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["IDDObstacleTrack", "IDDAnomalyTrack"],
        },
        "SegEval-TooSmall": {
            "file_fmt" : "SegEval-TooSmallResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["IDDObstacleTrack", "IDDAnomalyTrack"],
        },
        "SegEval-Small": {
            "file_fmt" : "SegEval-SmallResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["IDDObstacleTrack", "IDDAnomalyTrack"],
        },
        "SegEval-Large": {
            "file_fmt" : "SegEval-LargeResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["IDDObstacleTrack", "IDDAnomalyTrack"],
        },
        "SegEval-VeryLarge": {
            "file_fmt" : "SegEval-VeryLargeResults_{method_name}_{dataset_name}.hdf5",
            "valid_datasets": ["IDDObstacleTrack", "IDDAnomalyTrack"],
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
    result_root_dir = "./outputs/"
    experiments = ["PixBinaryClass", "SegEval", "SegEval-TooSmall", "SegEval-Small", "SegEval-Large", "SegEval-VeryLarge", "IntersectionOverUnion"]

    dataset_names = [
            "IDDObstacleTrack-static", 
            "IDDObstacleTrack-staticNormal", 
            "IDDObstacleTrack-staticLowLight",

            "IDDObstacleTrack-temporal",
            "IDDObstacleTrack-temporalNormal",
            "IDDObstacleTrack-temporalLowLight",

            "IDDAnomalyTrack-static",
            "IDDAnomalyTrack-staticNormal",
            "IDDAnomalyTrack-staticLowLight",

            "IDDAnomalyTrack-temporal",
            "IDDAnomalyTrack-temporalNormal",
            "IDDAnomalyTrack-temporalLowLight"
    ]

    method_names = ["PixOOD_IDD_RA"] 

    tables = {} 
    for dataset_name in dataset_names:
        data = []
        for method_name in method_names:
            res_dict = {}
            for exp in experiments:
                if dataset_name.split('-')[0] not in experiments_info[exp]["valid_datasets"]:
                    continue

                res_dict.update(get_results_for_exp(exp, method_name, dataset_name))

            data.append(Series(res_dict, name=method_name))

        tables[dataset_name] = DataFrame(data=data)
    with open(os.path.join(result_root_dir, "results.md"), 'w') as f:
        for dataset_name in dataset_names:
            f.write(f"===== {dataset_name} =====\n")
            f.write(tables[dataset_name].to_markdown(floatfmt=".2f"))
            f.write("\n")
            f.write("\n")
