
from pathlib import Path
from os import environ
from operator import itemgetter
import logging, re

from easydict import EasyDict
import numpy as np

from paths import DIR_DATASETS
from datasets.dataset_registry import DatasetRegistry
from datasets.dataset_io import DatasetBase, ChannelLoaderImage


log = logging.getLogger(__name__)

class DatasetIDD(DatasetBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.discover()

    def discover(self):
        """ Discover frames in file system """
        path_template = Path(self.channels['image'].resolve_template(
            dset = self,
            fid = '*',
        ))
        fids = [p.stem for p in path_template.parent.glob(path_template.name)]
        fids.sort()
        self.set_frames([EasyDict(fid=fid) for fid in fids])
        self.check_size()

    @staticmethod
    def mask_from_label_range(labels, id_or_range):
        if isinstance(id_or_range, (tuple, list)) and id_or_range.__len__() == 2:
            range_low, range_high = id_or_range
            return (range_low <= labels) & (labels <= range_high)
        else:
            return labels == id_or_range

    def get_frame(self, key, *channels):
        channels = set(channels)
        wants_labels_explicitly = False
        if 'label_pixel_gt' in channels:
            wants_labels_explicitly = True
            channels.remove('label_pixel_gt')
            channels.add('semantic_class_gt')

        fr = super().get_frame(key, *channels)

        sem_gt = fr.get('semantic_class_gt')
        if sem_gt is not None:
            h, w = sem_gt.shape[:2]
            label = np.full((h, w), 255, dtype=np.uint8)

            label[self.mask_from_label_range(sem_gt, self.cfg.classes.usual)] = 0
            label[self.mask_from_label_range(sem_gt, self.cfg.classes.anomaly)] = 1

            fr['label_pixel_gt'] = label
        elif wants_labels_explicitly:
            raise KeyError(f'No labels for {key} in {self}')

        return fr

    def __str__(self):
        dir_root = self.cfg.get('dir_root', 'NO DIR ROOT')
        return f'{self.cfg.name}({dir_root})'

    def set_frames(self, frame_list):
        """ Filter frames by requested scenes """
        frames_filtered = [
            fr for fr in frame_list
            if fr.fid.split('_')[0] in self.cfg.scenes
        ]
        super().set_frames(frames_filtered)


@DatasetRegistry.register_class()
class DatasetIDDAnomalyTrack(DatasetIDD):
    CLASS_IDS = dict(
        usual = [0, 18],
        anomaly = 254,
        ignore = 255,
    )

    DEFAULTS = dict(
        dir_root = DIR_DATASETS / 'dataset_IDDTrack',
        img_fmt = 'png',
        classes = CLASS_IDS,
        name_for_persistence = 'IDDAnomalyTrack-all',
    )

    SCENES_ALL = {
        'static',
        'temporal',
    }

    configs = [
        dict(
            # all
            name = 'IDDAnomalyTrack-all',
            scenes = SCENES_ALL,
            expected_length = 980+1138,
            **DEFAULTS,
        ),
        dict(
            # static
            name = 'IDDAnomalyTrack-static',
            scenes = {'static'},
            expected_length = 980,
            **DEFAULTS,
        ),
        dict(
            # temporal
            name = 'IDDAnomalyTrack-temporal',
            scenes ={'temporal'},
            expected_length = 1138,
            **DEFAULTS,
        ),
    ]

    channels = {
        'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.{dset.cfg.img_fmt}"),
        'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}.png"),
    }


@DatasetRegistry.register_class()
class DatasetIDDObstacleTrack(DatasetIDD):
    CLASS_IDS = dict(
        usual = 0,
        anomaly = 254,
        ignore = (np.arange(18)+1).tolist() + [255],
    )

    DEFAULTS = dict(
        dir_root = DIR_DATASETS / 'dataset_IDDTrack',
        img_fmt = 'png',
        classes = CLASS_IDS,
        name_for_persistence = 'IDDObstacleTrack-all',
    )

    SCENES_ALL = {
        'static',
        'temporal',
    }

    configs = [
        dict(
            # all
            name = 'IDDObstacleTrack-all',
            scenes = SCENES_ALL,
            expected_length = 980+1138,
            **DEFAULTS,
        ),
        dict(
            # static
            name = 'IDDObstacleTrack-static',
            scenes = {'static'},
            expected_length = 980,
            **DEFAULTS,
        ),
        dict(
            # temporal
            name = 'IDDObstacleTrack-temporal',
            scenes ={'temporal'},
            expected_length = 1138,
            **DEFAULTS,
        ),
    ]

    channels = {
        'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.{dset.cfg.img_fmt}"),
        'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}.png"),
    }

