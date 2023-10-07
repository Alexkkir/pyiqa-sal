from PIL import Image

import torch
from torch.utils import data as data
import torchvision.transforms as tf

from pyiqa.data.data_util import read_meta_info_file 
from pyiqa.data.transforms import transform_mapping, PairedToTensor
from pyiqa.utils.registry import DATASET_REGISTRY

from .base_iqa_dataset import BaseIQADataset
from .general_fr_dataset import GeneralFRDataset

@DATASET_REGISTRY.register()
class GeneralSalDataset(BaseIQADataset):
    """General NR dataset with saliency and meta info file.
    """
    
    def init_path_mos(self, opt):
        target_img_folder = opt['dataroot_target']
        sal_img_folder = opt['dataroot_sal']
        self.paths_mos = read_meta_info_file(target_img_folder, opt['meta_info_file']) 
        self.paths_saliency = read_meta_info_file(sal_img_folder, opt['meta_info_file']) 

    get_transforms = GeneralFRDataset.get_transforms

    def __getitem__(self, index):

        sal_path = self.paths_saliency[index][0]
        img_path = self.paths_mos[index][0]
        mos_label = self.paths_mos[index][1]
        img_pil = Image.open(img_path).convert('RGB')
        sal_pil = Image.open(sal_path).convert('RGB')
        
        img_pil, sal_pil = self.paired_trans([img_pil, sal_pil])

        img_tensor = self.common_trans(img_pil) * self.img_range
        sal_tensor = self.common_trans(sal_pil) * self.img_range
        mos_label_tensor = torch.Tensor([mos_label])
        
        return {'img': img_tensor, 'sal_img': sal_tensor, 'mos_label': mos_label_tensor, 'img_path': img_path, 'sal_img_path': sal_path}
