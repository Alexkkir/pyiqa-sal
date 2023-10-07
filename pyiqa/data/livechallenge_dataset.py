import os

from pyiqa.data.data_util import read_meta_info_file
from pyiqa.utils.registry import DATASET_REGISTRY
# from .general_nr_dataset import GeneralNRDataset
from .general_sal_dataset import GeneralSalDataset


@DATASET_REGISTRY.register()
class LIVEChallengeDataset(GeneralSalDataset):
    """The LIVE Challenge Dataset introduced by

    D. Ghadiyaram and A.C. Bovik, 
    "Massive Online Crowdsourced Study of Subjective and Objective Picture Quality," 
    IEEE Transactions on Image Processing, 2016
    url: https://live.ece.utexas.edu/research/ChallengeDB/index.html 
    
    Args:
        opt (dict): Config for train datasets with the following keys:
            phase (str): 'train' or 'val'.
    """

    def init_path_mos(self, opt):
        super().init_path_mos(opt)
        target_img_folder = os.path.join(opt['dataroot_target'], 'Images')
        self.paths_mos = read_meta_info_file(target_img_folder, opt['meta_info_file']) 
        # remove first 7 training images as previous works
        self.paths_mos = self.paths_mos[7:]
        self.paths_saliency = self.paths_saliency[7:]
