
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

STATIC_NORMAL = ["0", "1", "100", "101", "102", "104", "108", "11", "110",
                 "116", "121", "122", "125", "126", "127", "130", "131",
                 "141", "145", "146", "151", "152", "154", "155", "158",
                 "160", "162", "163", "167", "17", "171", "172", "173",
                 "174", "175", "176", "177", "178", "18", "181", "2", "20",
                 "204", "21", "211", "213", "214", "223", "23", "230",
                 "235", "236", "237", "24", "240", "242", "243", "245",
                 "246", "25", "252", "262", "263", "267", "272", "273",
                 "275", "276", "277", "28", "280", "282", "283", "284",
                 "285", "29", "294", "295", "3", "30", "308", "31", "311",
                 "314", "316", "317", "318", "319", "320", "322", "338",
                 "34", "340", "347", "350", "353", "355", "356", "357",
                 "359", "36", "360", "361", "37", "372", "375", "377",
                 "379", "38", "380", "385", "387", "388", "40", "400",
                 "403", "41", "412", "414", "416", "417", "42", "420",
                 "421", "422", "43", "431", "432", "436", "439", "44",
                 "441", "446", "45", "452", "453", "457", "459", "46",
                 "460", "467", "470", "472", "476", "48", "483", "486",
                 "489", "49", "490", "493", "495", "497", "498", "5",
                 "501", "502", "503", "508", "51", "512", "516", "518",
                 "52", "53", "532", "540", "541", "544", "548", "550",
                 "554", "56", "57", "578", "59", "60", "62", "63", "65",
                 "68", "69", "70", "77", "78", "80", "81", "84", "85",
                 "86", "87", "89", "91", "93", "94", "96", "98"]
STATIC_LOWLIGHT = ["rain", "fog", "lowlight"]
TEMPORAL_NORMAL = ["0", "1", "10", "100", "101", "102", "11", "12", "13",
   "14", "16", "18", "19", "2", "21", "22", "23", "24", "26", "27", "28",
                   "29", "3", "30", "34", "35", "36", "37", "4", "43",
                   "44", "46", "47", "48", "49", "50", "51", "53", "55",
                   "56", "57", "58", "59", "6", "60", "61", "62", "63",
                   "64", "65", "66", "67", "68", "69", "7", "70", "71",
                   "73", "74", "75", "76", "77", "78", "79", "8", "81",
                   "82", "84", "86", "87", "89", "9", "90", "92", "95",
                   "97", "98", "99"]
TEMPORAL_LOWLIGHT = ["25", "17", "45", "91", "41", "40", "33", "38", "52",
                     "39", "72", "93", "94", "31", "54", "15", "96", "88",
                     "85", "32", "20", "5", "42", "80", "83"]

VP0= ['0', '100', '101', '102', '12', '15', '18', '20', '21', '22', '23', '26',
      '28', '29', '3', '30', '31', '32', '34', '35', '36', '37', '39', '43',
      '44', '49', '50', '52', '53', '54', '57', '59', '60', '63', '66', '70',
      '71', '72', '73', '74', '75', '77', '8', '81', '83', '84', '85', '87',
      '88', '89', '9', '90', '92', '93', '94', '95', '96', '97', '98', '99']
VP1= ['1', '10', '13', '16', '2', '27', '42', '46', '47', '48', '5', '55',
      '56', '58', '64', '65', '67', '68', '69', '7', '76', '78', '80', '82', '86']
VP2= ['11', '14', '24', '51', '61', '62', '79']
VP3= ['17', '19', '25', '33', '38', '4', '40', '41', '45', '6', '91']


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
            label_all = np.full((h, w), 255, dtype=np.uint8)

            label[self.mask_from_label_range(sem_gt, self.cfg.classes.usual)] = 0
            label[self.mask_from_label_range(sem_gt, self.cfg.classes.anomaly)] = 1

            mask_usual = self.mask_from_label_range(sem_gt, self.cfg.classes.usual)
            label_all[mask_usual] = sem_gt[mask_usual]
            if isinstance(self.cfg.classes.usual, int):
                label_all[self.mask_from_label_range(sem_gt, self.cfg.classes.anomaly)] = self.cfg.classes.usual + 1
            else:
                label_all[self.mask_from_label_range(sem_gt, self.cfg.classes.anomaly)] = max(self.cfg.classes.usual) + 1
            fr['label_pixel_gt'] = label
            fr['label_pixel_gt_kp1'] = label_all
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
            if (fr.fid.split('_')[0] in self.cfg.scenes) and (fr.fid.split('_')[1] not in self.cfg.exclude)
        ]
        
        if hasattr(self.cfg, "include"):
            frames_filtered = [
                fr for fr in frames_filtered
                if (fr.fid.split('_')[1] in self.cfg.include)
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
        dir_root = DIR_DATASETS / 'dataset_ISSUTrack',
        img_fmt = 'png',
        classes = CLASS_IDS,
        name_for_persistence = 'ISSUAnomalyTrack-all',
    )

    SCENES_ALL = {
        'static',
        'staticNormal',
        'staticLowLight',
        'temporal',
        'temporalNormal',
        'temporalLowLight',
        'temporal_vp0',
        'temporal_vp1',
        'temporal_vp2',
        'temporal_vp3',
    }

    configs = [
        dict(
            # all
            name = 'ISSUAnomalyTrack-all',
            scenes = SCENES_ALL,
            expected_length = 980+1138,
            exclude = [], 
            **DEFAULTS,
        ),
        dict(
            # static
            name = 'ISSUAnomalyTrack-static',
            scenes = {'static'},
            expected_length = 980,
            exclude = [], 
            **DEFAULTS,
        ),
        dict(
            # static - Normal (nice) images only
            name = 'ISSUAnomalyTrack-staticNormal',
            scenes = {'static'},
            expected_length = 980-132,
            exclude = STATIC_LOWLIGHT, 
            **DEFAULTS,
        ),
        dict(
            # static - LowLight (rain, fog, ...)
            name = 'ISSUAnomalyTrack-staticLowLight',
            scenes = {'static'},
            expected_length = 132,
            exclude = STATIC_NORMAL,
            **DEFAULTS,
        ),
        dict(
            # temporal
            name = 'ISSUAnomalyTrack-temporal',
            scenes ={'temporal'},
            expected_length = 1138,
            exclude = [],
            **DEFAULTS,
        ),
        dict(
            # temporal - Normal
            name = 'ISSUAnomalyTrack-temporalNormal',
            scenes ={'temporal'},
            expected_length = 1138 - 270,
            exclude = TEMPORAL_LOWLIGHT,
            **DEFAULTS,
        ),
        dict(
            # temporal - LowLight
            name = 'ISSUAnomalyTrack-temporalLowLight',
            scenes ={'temporal'},
            expected_length = 270, 
            exclude = TEMPORAL_NORMAL,
            **DEFAULTS,
        ),
        dict(
            # temporal
            name = 'ISSUAnomalyTrack-temporalVP0',
            scenes ={'temporal'},
            expected_length = 675,
            exclude = [],
            include = VP0,
            **DEFAULTS,
        ),
        dict(
            # temporal
            name = 'ISSUAnomalyTrack-temporalVP1',
            scenes ={'temporal'},
            expected_length = 269,
            exclude = [],
            include = VP1,
            **DEFAULTS,
        ),
        dict(
            # temporal
            name = 'ISSUAnomalyTrack-temporalVP2',
            scenes ={'temporal'},
            expected_length = 72,
            exclude = [],
            include = VP2,
            **DEFAULTS,
        ),
        dict(
            # temporal
            name = 'ISSUAnomalyTrack-temporalVP3',
            scenes ={'temporal'},
            expected_length = 122,
            exclude = [],
            include = VP3,
            **DEFAULTS,
        )
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
        dir_root = DIR_DATASETS / 'dataset_ISSUTrack',
        img_fmt = 'png',
        classes = CLASS_IDS,
        name_for_persistence = 'ISSUObstacleTrack-all',
    )

    SCENES_ALL = {
        'static',
        'staticNormal',
        'staticLowLight',
        'temporal',
        'temporalNormal',
        'temporalLowLight',
    }

    configs = [
        dict(
            # all
            name = 'ISSUObstacleTrack-all',
            scenes = SCENES_ALL,
            expected_length = 980+1138,
            exclude = [], 
            **DEFAULTS,
        ),
        dict(
            # static
            name = 'ISSUObstacleTrack-static',
            scenes = {'static'},
            expected_length = 980,
            exclude = [], 
            **DEFAULTS,
        ),
        dict(
            # static - Normal (nice) images only
            name = 'ISSUObstacleTrack-staticNormal',
            scenes = {'static'},
            expected_length = 980-132,
            exclude = STATIC_LOWLIGHT, 
            **DEFAULTS,
        ),
        dict(
            # static - LowLight (rain, fog, ...)
            name = 'ISSUObstacleTrack-staticLowLight',
            scenes = {'static'},
            expected_length = 132,
            exclude = STATIC_NORMAL, 
            **DEFAULTS,
        ),
        dict(
            # temporal
            name = 'ISSUObstacleTrack-temporal',
            scenes ={'temporal'},
            expected_length = 1138,
            exclude = [],
            **DEFAULTS,
        ),
        dict(
            # temporal - Normal
            name = 'ISSUObstacleTrack-temporalNormal',
            scenes ={'temporal'},
            expected_length = 1138 - 270,
            exclude = TEMPORAL_LOWLIGHT, 
            **DEFAULTS,
        ),
        dict(
            # temporal - LowLight
            name = 'ISSUObstacleTrack-temporalLowLight',
            scenes ={'temporal'},
            expected_length = 270, 
            exclude = TEMPORAL_NORMAL,
            **DEFAULTS,
        )
    ]

    channels = {
        'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.{dset.cfg.img_fmt}"),
        'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}.png"),
    }

