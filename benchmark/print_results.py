import os
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from pandas import DataFrame, Series

from datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file


if __name__ == "__main__":
    result_root_dir = "./outputs/"
    experiments = ["PixBinaryClass", "SegEval-ObstacleTrack", "IntersectionOverUnion"]
    experiments_info = {
            "IntersectionOverUnion" : {
                "file_fmt" : "PixClassCurve_{method_name}_{dataset_name}.hdf5",
                "valid_datasets": ["IDDAnomalyTrack-static", "IDDAnomalyTrack-temporal"],
            },
            "PixBinaryClass": {
                "file_fmt" : "PixClassCurve_{method_name}_{dataset_name}.hdf5",
                "valid_datasets": ["IDDObstacleTrack-static", "IDDObstacleTrack-temporal", "IDDAnomalyTrack-static", "IDDAnomalyTrack-temporal"],
            },
            "SegEval-ObstacleTrack": {
                "file_fmt" : "SegEval-ObstacleTrackResults_{method_name}_{dataset_name}.hdf5",
                "valid_datasets": ["IDDObstacleTrack-static", "IDDObstacleTrack-temporal", "IDDAnomalyTrack-static", "IDDAnomalyTrack-temporal"],
            },
    }

    dataset_names = ["IDDObstacleTrack-static", "IDDObstacleTrack-temporal", "IDDAnomalyTrack-static", "IDDAnomalyTrack-temporal"]
    method_names = ["JSRNet_cs", "JSRNet_IDD", "DaCUP_cs", "DaCUP_IDD", "PixOOD_cs_RO", "PixOOD_cs_RA", "PixOOD_IDD_RO", "PixOOD_IDD_RA"] 

    tables = {} 
    for dataset_name in dataset_names:
        data = []
        for method_name in method_names:
            res_dict = {}
            for exp in experiments:
                if dataset_name not in experiments_info[exp]["valid_datasets"]:
                    continue

                dataset_name_tmp = dataset_name
                if exp == "IntersectionOverUnion":
                    dataset_name_tmp = "IDDAnomalyFullTrack-" + dataset_name.split('-')[1]

                filename = os.path.join(result_root_dir, exp, "data", experiments_info[exp]["file_fmt"].format(method_name=method_name, dataset_name=dataset_name_tmp))
                if not os.path.isfile(filename):
                    continue
                tmp_data = hdf5_read_hierarchy_from_file(filename)

                if exp == "PixBinaryClass":
                    res_dict["AP"] = 100*tmp_data.area_PRC 
                    res_dict["FPR@95TPR"] = 100*tmp_data.tpr95_fpr 
                elif exp == "SegEval-ObstacleTrack":
                    res_dict["mF1"] = 100*tmp_data.f1_mean 
                elif exp == "IntersectionOverUnion":
                    res_dict["closed-mIoU"] = 100*tmp_data.closed_miou 
                    res_dict["open-mIoU@best_f1"] = 100*tmp_data.open_miou["best_f1"]
                    res_dict["open-mIoU@tpr95"] = 100*tmp_data.open_miou["tpr95"]
                    res_dict["open-mIoU@fpr05"] = 100*tmp_data.open_miou["fpr05"]

            data.append(Series(res_dict, name=method_name))

        tables[dataset_name] = DataFrame(data=data)
    with open(os.path.join(result_root_dir, "results.md"), 'w') as f:
        for dataset_name in dataset_names:
            f.write(f"===== {dataset_name} =====\n")
            f.write(tables[dataset_name].to_markdown(floatfmt=".2f"))
            f.write("\n")
            f.write("\n")
