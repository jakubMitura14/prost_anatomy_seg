import glob
import monai
import os
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
import functools
import multiprocessing as mp
import os
from functools import partial
import numpy as np
import pandas as pd
from toolz.itertoolz import groupby
import SimpleITK as sitk
import os
import shutil

root_dir='/home/sliceruser/nnunetMainFolder/nnUNet_raw/Dataset111_Prostate/imagesTr'
def group_files(root_dir):
    """ 
    based on name will group the files into triplets of related files
    """
    all_files=os.listdir(root_dir)
    grouped_by_master= groupby(lambda path : path.split('_')[0],all_files)
    return list(dict(grouped_by_master).items())

grouped=group_files(root_dir)   
grouped= list(map(lambda el: el[1],grouped)) 
grouped = np.sort(np.array(grouped))
# grouped = np.array(grouped)

t2w_path='/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-t2-prostate-mri'
adc_path= '/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-adc-prostate-mri'
hbv_path='/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-hbv-prostate-mri'


def my_file_copy(root_old,root_new,file_name,strr):
    inputImageFileName=f"{root_old}/{file_name}"
    image = sitk.ReadImage(inputImageFileName)
    file_name= file_name.replace('.nii.gz',f"_{strr}.mha")
    outputImageFileName=f"{root_new}/{file_name}"
    sitk.WriteImage(image, outputImageFileName)



list(map( lambda p: my_file_copy(root_dir,t2w_path,p[0],'t2w'),grouped ))
list(map( lambda p: my_file_copy(root_dir,adc_path,p[1],'adc'),grouped ))
list(map( lambda p: my_file_copy(root_dir,hbv_path,p[2],'hbv'),grouped ))

print(grouped)
"""
docker run --cpus=8 --memory=12gb --shm-size=12gb --gpus='"device=0"' -it --rm \
    -v /media/jm/hddData/datasets/for_pic/data/for_picai_docker/test/:/input \
    -v /media/jm/hddData/datasets/for_pic/data/picai_output/:/output \
    joeranbosma/picai_prostate_segmentation_processor


    docker run --cpus=8 --memory=12gb --shm-size=12gb --gpus='"device=0"' -it --privileged --rm \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/:/input \
    -v /workspaces/prost_anatomy_seg/data/picai_output/:/output \
    joeranbosma/picai_prostate_segmentation_processor


        -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-t2-prostate-mri:/input/images/transverse-t2-prostate-mri \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/ttransverse-hbv-prostate-mri:/input/images/transverse-hbv-prostate-mri \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-adc-prostate-mri:/input/images/transverse-adc-prostate-mri \

    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-t2-prostate-mri/9000100_0000_t2w.mha:/input/images/transverse-t2-prostate-mri/9000100_0000_t2w.mha \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/ttransverse-hbv-prostate-mri/9000100_0002_hbv.mha:/input/images/transverse-hbv-prostate-mri/9000100_0002_hbv.mha \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-adc-prostate-mri/9000100_0001_adc.mha:/input/images/transverse-adc-prostate-mri/9000100_0001_adc.mha \


    --mount type=bind,source="/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-t2-prostate-mri/9000100_0000_t2w.mha",target="/input/images/transverse-t2-prostate-mri/9000100_0000_t2w.mha" \
    --mount type=bind,source="/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/ttransverse-hbv-prostate-mri/9000100_0002_hbv.mha",target="/input/images/transverse-hbv-prostate-mri/9000100_0002_hbv.mha" \
    --mount type=bind,source="/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-adc-prostate-mri/9000100_0001_adc.mha",target="/input/images/transverse-adc-prostate-mri/9000100_0001_adc.mha" \



        -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-t2-prostate-mri:/input/images/transverse-t2-prostate-mri \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/ttransverse-hbv-prostate-mri:/input/images/transverse-hbv-prostate-mri \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-adc-prostate-mri:/input/images/transverse-adc-prostate-mri \

         -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-t2-prostate-mri/9000100_0000_t2w.mha:/input/images/transverse-t2-prostate-mri/9000100_0000_t2w.mha:ro \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/ttransverse-hbv-prostate-mri/9000100_0002_hbv.mha:/input/images/transverse-hbv-prostate-mri/9000100_0002_hbv.mha:ro \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-adc-prostate-mri/9000100_0001_adc.mha:/input/images/transverse-adc-prostate-mri/9000100_0001_adc.mha:ro \
      joeranbosma/picai_prostate_segmentation_processor

        
    --mount type=bind,source=

    docker run --cpus=8 --memory=12gb --shm-size=12gb -it -i --privileged --rm \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test:/input \
    -v $HOME/data/picai_output/images/transverse-whole-prostate-mri:/output/images/transverse-whole-prostate-mri \
        joeranbosma/picai_prostate_segmentation_processor

    
            """

$HOME/data/picai_output

/workspaces/prost_anatomy_seg/data/picai_output/images/transverse-whole-prostate-mri

sudo chmod -R 777 /workspaces/prost_anatomy_seg/data/picai_output
sudo chmod -R a+rwX /workspaces/prost_anatomy_seg/data/picai_output
sudo chown -R :users /workspaces/prost_anatomy_seg/data/picai_output

/images/transverse-whole-prostate-mri
/images/transverse-whole-prostate-mri


    docker run --cpus=8 --memory=12gb --shm-size=12gb -it -i --privileged --rm \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test:/input \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-t2-prostate-mri:/input/images/transverse-t2-prostate-mri \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/ttransverse-hbv-prostate-mri:/input/images/transverse-hbv-prostate-mri \
    -v /workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-adc-prostate-mri:/input/images/transverse-adc-prostate-mri \
    -v $HOME/data/picai_output/images/transverse-whole-prostate-mri:/output/images/transverse-whole-prostate-mri \
    --mount type=bind,source="/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-t2-prostate-mri/9000100_0000_t2w.mha",target="/input/images/transverse-t2-prostate-mri/9000100_0000_t2w.mha" \
    --mount type=bind,source="/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/ttransverse-hbv-prostate-mri/9000100_0002_hbv.mha",target="/input/images/transverse-hbv-prostate-mri/9000100_0002_hbv.mha" \
    --mount type=bind,source="/workspaces/prost_anatomy_seg/data/for_picai_docker/test/images/transverse-adc-prostate-mri/9000100_0001_adc.mha",target="/input/images/transverse-adc-prostate-mri/9000100_0001_adc.mha" \   
      joeranbosma/picai_prostate_segmentation_processor


chmod -R 777 /workspaces/prost_anatomy_seg/data/for_picai_docker/test