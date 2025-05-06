import dataclasses
from typing import List, Dict
import os

from pathlib import Path
from pandas import DataFrame, Series
import numpy as np
from easydict import EasyDict
import torch
from torchmetrics.functional.classification import multiclass_confusion_matrix
from metrics.base import EvaluationMetric, MetricRegistry, save_table


from paths import DIR_OUTPUTS
from datasets.utils import adapt_img_data, get_heat, imwrite
from datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file
from datasets.utils import adapt_img_data, imwrite
from metrics.base import save_table


def compute_confusion_matrix(pred, actual, mask, num_classes):
    pred_in_roi = torch.tensor(pred[mask])
    actual_in_roi = torch.tensor(actual[mask])
    cm = multiclass_confusion_matrix(pred_in_roi, actual_in_roi, num_classes=num_classes)
    return cm


@dataclasses.dataclass
class RecResultsInfo:
    method_name : str
    dataset_name : str

    threshold_types : np.ndarray 
    threshold : np.ndarray 
    open_miou : Dict
    closed_miou : float 

    def __iter__(self):
        return dataclasses.asdict(self).items()

    def save(self, path):
        hdf5_write_hierarchy_to_file(path, dataclasses.asdict(self))

    @classmethod
    def from_file(cls, path):
        return cls(**hdf5_read_hierarchy_from_file(path))


@MetricRegistry.register_class()
class MetricPixelRecognition(EvaluationMetric):

    configs = [
        EasyDict(
            name = 'IntersectionOverUnion',
            num_classes = 19,
            threshold_types = ["tpr95", "fpr05", "best_f1"] + [f"{int(100*i)}" for i in np.linspace(0.1, 0.9, num=9)] + ["99"],
            threshold = None, # should be set dynamically
        ),
    ]

    @property
    def name(self):
        return self.cfg.name
    
    @staticmethod
    def vis_frame(fid, dset_name, method_name, mask_roi, anomaly_p, image = None, label_pixel_gt = None, **_):
        h, w = mask_roi.shape[:2]

        canvas = image.copy() if image is not None else np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_color = adapt_img_data(anomaly_p)
        canvas[mask_roi] = canvas[mask_roi]//2 + heatmap_color[mask_roi]//2
        imwrite(
            DIR_OUTPUTS / f'vis_OpensetClassification' / method_name / dset_name / f'{fid}_demo_anomalyP.webp',
            canvas,
        )

        anomaly_heat = get_heat(anomaly_p, overlay=label_pixel_gt)
        imwrite(
            DIR_OUTPUTS / f'vis_OpensetClassification' / method_name / dset_name / f'{fid}_demo_anomalyP_heat.webp',
            anomaly_heat,
        )


    def process_frame(self, label_pixel_gt_kp1 : np.ndarray, anomaly_p : np.ndarray, class_p : np.ndarray, fid : str=None, dset_name : str=None, method_name : str=None, visualize : bool = True, **_):
        """
        @param label_pixel_gt_kp1: HxW uint8
            [0, X] = usual classes
            X+1 = obstacle
            255 = ignore
        @param anomaly_p: HxW float16
            heatmap of per-pixel anomaly detection, value from 0 to 1
        @param class_p: HxW float16
            closed-set predictions over K classes
        @param fid: frame identifier, for saving extra outputs
        @param dset_name: dataset identifier, for saving extra outputs
        """
        try:
            mask_roi_open_set = label_pixel_gt_kp1 < 255
            mask_roi_closed_set = label_pixel_gt_kp1 < self.cfg.num_classes
        except TypeError:
            raise RuntimeError(f"No ground truth available for {fid}. Please check dataset path...")

        closed_set_confusion_matrix = compute_confusion_matrix(
            class_p, label_pixel_gt_kp1, mask_roi_closed_set, self.cfg.num_classes
        )

        # visualization
        if visualize and fid is not None and dset_name is not None and method_name is not None:
            self.vis_frame(fid=fid, dset_name=dset_name, method_name=method_name, mask_roi=mask_roi_open_set,
                           anomaly_p=anomaly_p, label_pixel_gt=label_pixel_gt_kp1, **_)
        #print('Vrange', np.min(predictions_in_roi), np.mean(predictions_in_roi), np.max(predictions_in_roi))

        open_set_confusion_matrix = {}
        for i in range(0, len(self.cfg.threshold)):
            class_p_thr = class_p.copy()
            class_p_thr[anomaly_p > self.cfg.threshold[i]] = self.cfg.num_classes
            open_set_confusion_matrix[self.cfg.threshold_types[i]] = compute_confusion_matrix(
                class_p_thr, label_pixel_gt_kp1, mask_roi_open_set, self.cfg.num_classes + 1)

        return EasyDict(
            closed_set = closed_set_confusion_matrix,
            open_set = open_set_confusion_matrix,
        )


    def aggregate(self, frame_results : list, method_name : str, dataset_name : str):
        closed_set_cmats = [result.closed_set for result in frame_results]
        closed_set_cmat = sum(closed_set_cmats)
        closed_set_ious = torch.diag(closed_set_cmat) / (closed_set_cmat.sum(dim=0) + closed_set_cmat.sum(dim=1) - torch.diag(closed_set_cmat))
        miou = torch.nansum(closed_set_ious) / self.cfg.num_classes

        open_miou = {}
        for i in range(0, len(self.cfg.threshold)):
            open_set_cmats = [result.open_set[self.cfg.threshold_types[i]] for result in frame_results]
            open_set_cmat = sum(open_set_cmats)
            open_set_ious = torch.diag(open_set_cmat) / (open_set_cmat.sum(0) + open_set_cmat.sum(1) - torch.diag(open_set_cmat))

            open_miou[self.cfg.threshold_types[i]] = (np.nansum(open_set_ious[:-1]) / self.cfg.num_classes).item()

        out = EasyDict(
            closed_set_ious = closed_set_ious,
            open_set_ious = open_set_ious[:-1],
            closed_miou = miou.item(),
            open_miou = open_miou,
            method_name = method_name,
            dataset_name = dataset_name,
        )
        return out

    def persistence_path_data(self, method_name, dataset_name):
        return DIR_OUTPUTS / self.name / 'data' / f'OpenSet_{method_name}_{dataset_name}.hdf5'

    def persistence_path_plot(self, comparison_name, plot_name):
        return DIR_OUTPUTS / self.name / 'plot' / f'{comparison_name}__{plot_name}'

    def save(self, aggregated_result, method_name : str, dataset_name : str, path_override : Path = None):
        out_path = path_override or self.persistence_path_data(method_name, dataset_name)
        RecResultsInfo(method_name = method_name,
                       dataset_name = dataset_name,
                       threshold_types = np.array(self.cfg.threshold_types, dtype='S'),
                       threshold = np.array(self.cfg.threshold),
                       open_miou = aggregated_result.open_miou,
                       closed_miou = aggregated_result.closed_miou
                       ).save(out_path)

    def load(self, method_name : str, dataset_name : str, path_override : Path = None):
        pass

    def fields_for_table(self):
        return ['open-mIoU', 'closed-mIoU']

    def plot_many(self, aggregated_results : List, comparison_name : str, close : bool = True, method_names={}, plot_formats={}):
        table = DataFrame(data=[
            Series({**{f"open_miou@{k}": v for k, v in crv.open_miou.items()}, 'closed_miou': crv.closed_miou}, name=crv.method_name)
            for crv in aggregated_results
        ])
        # print(table)
        save_table(self.persistence_path_plot(comparison_name, 'OpensetTable'), table)

    def init(self, method_name, dataset_name):
        self.get_thresh_p_from_curve(method_name, dataset_name)

    def get_thresh_p_from_curve(self, method_name, dataset_name):
        # if "IDDAnomaly" in _dataset_name:
        #     dataset_name = "IDDAnomalyTrack-" + _dataset_name.split('-')[-1]
        # else:
        #     dataset_name = "IDDObstacleTrack-" + _dataset_name.split('-')[-1]

        out_path = DIR_OUTPUTS / "PixBinaryClass" / 'data' / f'PixClassCurve_{method_name}_{dataset_name}.hdf5'
        assert os.path.isfile(out_path), f"To load dynamically thresholds ({out_path}) for the open-set evaluation, run evaluation on {dataset_name} first!"
        pixel_results = hdf5_read_hierarchy_from_file(out_path)
        self.cfg.threshold = [pixel_results.tpr95_threshold, pixel_results.fpr5_threshold, pixel_results.best_f1_threshold] + np.linspace(0.1, 0.9, num=9).tolist() + [0.99]

        return self.cfg.threshold
