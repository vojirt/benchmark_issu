
from typing import List
from pathlib import Path
from pandas import DataFrame, Series
import numpy as np
from matplotlib import pyplot
from easydict import EasyDict

from metrics.base import EvaluationMetric, MetricRegistry, save_figure, save_table
from metrics.pixel_classification_curves import BinaryClassificationCurve, curves_from_cmats, plot_classification_curves, reduce_curve_resolution

from paths import DIR_OUTPUTS
from datasets.utils import adapt_img_data, get_heat, imwrite
from metrics.base import save_table

def confusion_matrix(predicted : np.ndarray, target : np.ndarray, num_classes : int):
	x = predicted + num_classes * target
	bincount_2d = np.bincount(
		x.astype(np.int64), minlength=num_classes ** 2)
	assert bincount_2d.size == num_classes ** 2
	conf_mat = bincount_2d.reshape((num_classes, num_classes))
	return conf_mat

def compute_confusion_matrix(pred, actual, mask, num_classes):
	pred_in_roi = pred[mask]
	actual_in_roi = actual[mask]

	closed_set_cm = confusion_matrix(
		predicted=pred_in_roi,
		target=actual_in_roi,
		num_classes=num_classes,
	)
	return closed_set_cm

@MetricRegistry.register_class()
class MetricPixelRecognition(EvaluationMetric):

	configs = [
		EasyDict(
			name = 'IntersectionOverUnion',
			num_classes = 19,
			treshold = None, # should be set
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
			DIR_OUTPUTS / f'vis_PixelClassification' / method_name / dset_name / f'{fid}_demo_anomalyP.webp',
			canvas,
		)

		anomaly_heat = get_heat(anomaly_p, overlay=label_pixel_gt)
		imwrite(
			DIR_OUTPUTS / f'vis_PixelClassification' / method_name / dset_name / f'{fid}_demo_anomalyP_heat.webp',
			anomaly_heat,
		)


	def process_frame(self, label_pixel_gt : np.ndarray, anomaly_p : np.ndarray, class_p : np.ndarray, fid : str=None, dset_name : str=None, method_name : str=None, visualize : bool = True, **_):
		"""
		@param label_pixel_gt: HxW uint8
			0 = road
			1 = obstacle
			255 = ignore
		@param anomaly_p: HxW float16
			heatmap of per-pixel anomaly detection, value from 0 to 1
		@param class_p: HxW float16
			closed-set predictions over K classes
		@param fid: frame identifier, for saving extra outputs
		@param dset_name: dataset identifier, for saving extra outputs
		"""
		try:
			mask_roi_open_set = label_pixel_gt < 255
			mask_roi_closed_set = label_pixel_gt < 254
		except TypeError:
			raise RuntimeError(f"No ground truth available for {fid}. Please check dataset path...")

		closed_set_confusion_matrix = compute_confusion_matrix(
			class_p, label_pixel_gt, mask_roi_closed_set, self.cfg.num_classes)

		class_p[anomaly_p > self.cfg.treshold] = self.cfg.num_classes
		open_set_confusion_matrix = compute_confusion_matrix(
			class_p, label_pixel_gt, mask_roi_open_set, self.cfg.num_classes + 1)

		# visualization
		if visualize and fid is not None and dset_name is not None and method_name is not None:
			self.vis_frame(fid=fid, dset_name=dset_name, method_name=method_name, mask_roi=mask_roi_open_set,
						   anomaly_p=anomaly_p, label_pixel_gt=label_pixel_gt, **_)

		#print('Vrange', np.min(predictions_in_roi), np.mean(predictions_in_roi), np.max(predictions_in_roi))

		return EasyDict(
			closed_set = closed_set_confusion_matrix,
			open_set = open_set_confusion_matrix,
		)


	def aggregate(self, frame_results : list, method_name : str, dataset_name : str):
		closed_set_cmats = [result.closed_set for result in frame_results]
		closed_set_cmat = sum(closed_set_cmats)
		closed_set_ious = np.diag(closed_set_cmat) / (np.sum(closed_set_cmat, axis=0) + np.sum(closed_set_cmat, axis=1) - np.diag(closed_set_cmat))
		miou = np.nansum(closed_set_ious) / self.cfg.num_classes

		open_set_cmats = [result.open_set for result in frame_results]
		open_set_cmat = sum(open_set_cmats)
		open_set_ious = np.diag(open_set_cmat) / (np.sum(open_set_cmat, axis=0) + np.sum(open_set_cmat, axis=1) - np.diag(open_set_cmat))
		
		open_miou = np.nansum(open_set_ious[:-1]) / self.cfg.num_classes

		out = EasyDict(
			closed_set_ious = closed_set_ious,
			open_set_ious = open_set_ious[:-1],
			closed_miou = miou,
			open_miou = open_miou,
			method_name = method_name,
			dataset_name = dataset_name,
		)
		return out

	def persistence_path_data(self, method_name, dataset_name):
		return DIR_OUTPUTS / self.name / 'data' / f'PixClassCurve_{method_name}_{dataset_name}.hdf5'

	def persistence_path_plot(self, comparison_name, plot_name):
		return DIR_OUTPUTS / self.name / 'plot' / f'{comparison_name}__{plot_name}'

	def save(self, aggregated_result, method_name : str, dataset_name : str, path_override : Path = None):
		out_path = path_override or self.persistence_path_data(method_name, dataset_name)
		pass

	def load(self, method_name : str, dataset_name : str, path_override : Path = None):
		pass

	def fields_for_table(self):
		return ['open-mIoU', 'closed-mIoU']

	def plot_many(self, aggregated_results : List, comparison_name : str, close : bool = True, method_names={}, plot_formats={}):
		table = DataFrame(data=[
			Series({'open_miou': crv.open_miou, 'closed_miou': crv.closed_miou }, name=crv.method_name)
			for crv in aggregated_results
		])
		print(table)
		save_table(self.persistence_path_plot(comparison_name, 'PixClassTable'), table)
