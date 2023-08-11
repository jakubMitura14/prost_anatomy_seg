import os
import subprocess
from pathlib import Path

import SimpleITK as sitk
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      resample_to_reference_scan)
"""
segmenation where target is the sum of t2w,adc,hbv labels and we add as input additionally whole prostate segmentations

"""

import SimpleITK as sitk
import mdai
import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import multiprocessing as mp
import functools
from functools import partial
import mdai
import math
import time
import itertools
from pydicom.fileset import FileSet
from os import path as pathOs
from pathlib import Path
import toolz
from toolz.curried import pipe, map, filter, get
from toolz import curry
from os.path import basename, dirname, exists, isdir, join, split
import nnunetv2

import elastixRegister as elastixRegister
from elastixRegister import reg_a_to_b
from datetime import date
from toolz.itertoolz import groupby
from toolz import curry
# import multiprocess
# p = multiprocess.Pool(os.cpu_count())
import multiprocessing as mp
import json
import os
from subprocess import Popen
import subprocess
from prepareNNunet import *

resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'

sourceFrame = pd.read_csv(resCSVDir) 
cols=sourceFrame.columns
main_modality = 't2w'

# modalities that we want to include in the model
modalities_of_intrest=['t2w','adc','hbv']
prostate_col= 'pg_noSeg' # name of the column with segmentaton of whole prostate gland
#  'ob_noSeg' 'ob_Seg' 'ob_num' 'pg_noSeg'
#  'pg_Seg' 'pg_num' 'pz_noSeg' 'pz_Seg' 'pz_num' 'sv_l_noSeg' 'sv_l_Seg'
#  'sv_l_num' 'sv_r_noSeg' 'sv_r_Seg' 'sv_r_num' 'tz_noSeg' 'tz_Seg'
#  'tz_num' 'ur_noSeg' 'ur_Seg' 'ur_num'

non_mri_inputs=[]

label_cols=[prostate_col]
channel_names={  
    "0": "t2w", 
    "1": "adc",
    "2": "hbv",
    }
label_names= {  
    "background": 0,
    "prostate": 1,
    }
def process_labels_prim(labels,group,main_modality,label_new_path,zipped_modalit_path,out_pathsDict):
    copy_changing_type(labels[0], label_new_path)
    return [label_new_path],zipped_modalit_path



dataset_id='111'

grouped_rows= main_prepare_nnunet(dataset_id,modalities_of_intrest,channel_names,label_names,label_cols,process_labels_prim,non_mri_inputs,sourceFrame
                                  ,main_modality,is_test_prep=False,is_to_preprocess=True,generate_plans=False)
    

#mkdir -p /workspaces/prost_anatomy_seg/picai_ready/results/nnUNet/3d_fullres/Task2202_prostate_segmentation/nnUNet/3d_fullres/Task2202_prostate_segmentation/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1
# cp -a /workspaces/prost_anatomy_seg/picai_ready/results/nnUNet/3d_fullres/Task2202_prostate_segmentation/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/ /workspaces/prost_anatomy_seg/picai_ready/results/nnUNet/3d_fullres/Task2202_prostate_segmentation/nnUNet/3d_fullres/Task2202_prostate_segmentation/

#RESULTS_FOLDER="/workspaces/prost_anatomy_seg/picai_ready/results/nnUNet/3d_fullres/Task2202_prostate_segmentation"  nnUNet_predict -i "/home/sliceruser/nnunetMainFolder/nnUNet_raw/Dataset111_Prostate/imagesTr" -o "/workspaces/prost_anatomy_seg/data/picai_output"  -t Task2202_prostate_segmentation  -m '3d_fullres'  -tr nnUNetTrainerV2_Loss_FL_and_CE_checkpoints



# nnUNetv2_predict 
# -i /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset111_Prostate/imagesTr
# -o /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/my_prost_infered 
# -d 111 
# -c '3d_fullres' 
# -tr nnUNetTrainerV2_Loss_FL_and_CE_checkpoints


# Task2202_prostate_segmentation
# train_2_folds_and_save()

# nnUNetv2_find_best_configuration 280 -c CONFIGURATIONS 
# nnUNetv2_find_best_configuration 280 -c '3d_fullres' 

# # nnUNetv2_find_best_configuration 280 -c '3d_fullres' -f 0 -np 8
# # nnUNetv2_find_best_configuration -h
# #The file "/home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset280_Prostate/labelsTr/9043100.nii.gz" does not exist.


# --save_probabilities

# cp /home/sliceruser/workspaces/konwersjaJsonData/nnunetMainFolder/nnUNet_raw/Dataset280_Prostate/imagesTr/9000100_0003.nii.gz /workspaces/konwersjaJsonData/explore/in_prost_seg/9000100_0003.nii.gz

# cp -a /workspaces/konwersjaJsonData/explore/full_prostate_results/2023-05-08/Dataset280_Prostate /home/sliceruser/workspaces/konwersjaJsonData/nnUNet_results
# nnUNetv2_train 280 3d_fullres 0 -tr nnUNetTrainerV2_Loss_CE


# cp -a /workspaces/prost_anatomy_seg/picai_ready/results/nnUNet/3d_fullres/Task2202_prostate_segmentation/nnUNetTrainerV2_Loss_FL_and_CE_checkpoints__nnUNetPlansv2.1/ /home/sliceruser/nnUNet_results/Dataset111_Prostate