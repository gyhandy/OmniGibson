# Functionality imports
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import os

# OmniGibson imports
import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Open, OnTop, Inside, Folded, Unfolded, Overlaid
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import SemanticsAPI

# Utility imports
from IPython import embed
import matplotlib.pyplot as plt
from omni.isaac.synthetic_utils.visualization import colorize_bboxes
import json

# gm global vars
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True


def check_visible(seg_instance, obj_id, bboxes, threshold_ratio=0.5):
    '''
    Checks if all links (that have bbox we are interested in) are visible.
    Heuristic: object segmentation must cover at least half pixels in every link bbox
    NOW ASSUMES MINMAX BBOX!
    '''
    # print(len(bboxes))
    min_ratio = 1
    for bbox in bboxes:
        bbox_size = (bbox[3]-bbox[1])*(bbox[4]-bbox[2]) 
        if bbox_size == 0:
            continue
        # This is so xxxxxxx tricky. Need to select the y indicies first!
        pixel_count = np.sum(seg_instance[bbox[2]:bbox[4]+1, bbox[1]:bbox[3]+1] == obj_id)
        min_ratio = min(min_ratio, (pixel_count/bbox_size))
        # if pixel_count/bbox_size < threshold_ratio:
        #     return False
        # if min_ratio == 0:
        #     embed()
        
    return min_ratio
    