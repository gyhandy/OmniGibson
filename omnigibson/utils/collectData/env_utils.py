# Functionality imports
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import os

# OmniGibson imports
import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import (
    get_available_og_scenes,
    get_all_object_categories,
    get_og_avg_category_specs,
    get_object_models_of_category,
)
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Open, OnTop, Inside, Folded, Unfolded, Overlaid
import omnigibson.utils.transform_utils as T

# Utility imports
from IPython import embed
import matplotlib.pyplot as plt
from omni.isaac.synthetic_utils.visualization import colorize_bboxes
import json
from omnigibson.utils.collectData.openable_obj import (
    # get_openable_bottom_cabinet_train,
    # get_openable_microwave_train,
    # get_openable_bottom_cabinet_test,
    get_objects_by_categories,
)
# gm global vars
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True

gm.USE_GPU_DYNAMICS = True


def create_env_with_light():
    cfg = dict()
    cfg["scene"] = {
        "type": "Scene",
        "floor_plane_visible": True,
    }
    cfg['objects'] = [
        {
            "type": "LightObject",
            "name": "brilliant_light0",
            "light_type": "Sphere",
            "intensity": 40000,
            "radius": 0.1,
            "position": [3.0, -3.0, 4.0],
        },
        {
            "type": "LightObject",
            "name": "brilliant_light1",
            "light_type": "Sphere",
            "intensity": 40000,
            "radius": 0.1,
            "position": [-3.0, -3.0, 4.0],
        },
    ]
    env = og.Environment(cfg)
    for _ in range(30): env.step([])

    ########## Initiate camera ########################
    cam = VisionSensor(
            prim_path="/World/viewer_camera",
            name="camera",
            modalities=["rgb", "seg_instance","bbox_2d_loose", "bbox_2d_tight", "seg_semantic"], #"depth_linear", "seg_instance", "bbox_2d_tight", "bbox_3d", "camera"],
            image_height=1024,
            image_width=1024,
        )
    cam.initialize()
    # Allow camera teleoperation
    og.sim.enable_viewer_camera_teleoperation()

    return env, cam

def sample_cam_pose(yaw_low=-np.pi, yaw_high=0, dist_low=2.5, dist_high=4.5, pitch_low=-np.pi/4, pitch_high=np.pi/16):
    '''
    Helper function to saple a random camera pose.
    Later used to let cam focus on given object.
    '''
    camera_yaw = np.random.uniform(yaw_low, yaw_high)
    camera_dist = np.random.uniform(dist_low, dist_high)
    camera_pitch = np.random.uniform(low=pitch_low, high=pitch_high)
    target_to_camera = R.from_euler("yz", [camera_pitch, camera_yaw]).apply([1, 0, 0])
    raw_camera_pos = target_to_camera * camera_dist

    return raw_camera_pos

def fold_cloth(obj):
    def print_state():
        folded = obj.states[Folded].get_value()
        unfolded = obj.states[Unfolded].get_value()
        info = f"{obj.name}: [folded] %d [unfolded] %d" % (folded, unfolded)
        print(info)

    pos = obj.root_link.particle_positions
    x_min, x_max = np.min(pos, axis=0)[0], np.max(pos, axis=0)[0]
    x_extent = x_max - x_min
    # Get indices for the bottom 10 percent vertices in the x-axis
    indices = np.argsort(pos, axis=0)[:, 0][:(pos.shape[0] // 10)]
    start = np.copy(pos[indices])

    # lift up a bit
    mid = np.copy(start)
    mid[:, 2] += x_extent * 0.2

    # move towards x_max
    end = np.copy(mid)
    end[:, 0] += x_extent * 0.9

    increments = 25
    for ctrl_pts in np.concatenate([np.linspace(start, mid, increments), np.linspace(mid, end, increments)]):
        pos = obj.root_link.particle_positions
        pos[indices] = ctrl_pts
        obj.root_link.particle_positions = pos
        og.sim.step()
        print_state()

def basic_objects():
    cab0_name = f"cab_{0}"
    cab0 = DatasetObject(
        prim_path=f"/World/{cab0_name}",
        name=cab0_name,
        category="bottom_cabinet",
        # model="lwjdmj",
        model="ujniob",
    )

    cab1_name = f"cab_{1}"
    cab1 = DatasetObject(
        prim_path=f"/World/{cab1_name}",
        name=cab1_name,
        category="bottom_cabinet",
        model="ybntlp",
    )

    cupcake_name = "cake"
    cupcake = DatasetObject(
        prim_path=f"/World/{cupcake_name}",
        name=cupcake_name,
        category="cupcake",
        model="mbhweg",
        scale=2,
    )

    laptop = DatasetObject(
        prim_path="/World/laptop",
        name="laptop",
        category="laptop", 
        model="nvulcs",
        scale=0.2,
    )

    milk = DatasetObject(
        prim_path="/World/milk",
        name="milk",
        category="milk", 
        model="xbvdpc"
    )

    table = DatasetObject(
        prim_path="/World/table",
        name="breakfast_table",
        category="breakfast_table", 
        model="19203"
    )

    shirt = DatasetObject(
        prim_path="/World/shirt",
        name="shirt",
        category="t-shirt", 
        model="t-shirt_000", 
        prim_type=PrimType.CLOTH, 
        abilities={"foldable": {}, "unfoldable": {}},
        scale=0.05,
        position=[0, 0, 0.5],
        orientation=[0.7071, 0., 0.7071, 0.],
    )

    carpet = DatasetObject(
        prim_path="/World/carpet",
        name="carpet",
        category="carpet", 
        model="carpet_0", 
        prim_type=PrimType.CLOTH, 
        abilities={"foldable": {}, "unfoldable": {}},
        position=[0, 0, 0.5],
    )

    return cab0, cab1, cupcake, laptop, table, shirt, carpet