# Functionality imports
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import os

# OmniGibson imports
import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.objects import DatasetObject, LightObject
from omnigibson.utils.asset_utils import (
    get_available_og_scenes,
    get_available_g_scenes,
    get_all_object_categories,
    get_og_avg_category_specs,
    get_all_object_category_models,
)
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Open, OnTop, Inside, Folded, Unfolded, Overlaid, Filled
import omnigibson.utils.transform_utils as T
import trimesh

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

def create_predefined_env(scene_id=3, load_categories=None):
    scenes = get_available_og_scenes()
    scene_type = "InteractiveTraversableScene"
    scene_model = list(scenes)[scene_id]
    # print(scene_id)
    # exit(0)
    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
            "load_object_categories": ["floors", "walls", "ceilings", "top_cabinet", "countertop", "breakfast_table", "bottom_cabinet", "fridge", "microwave", "picture"], 
        },
    }
    if load_categories is not None:
        cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]+load_categories
    env = og.Environment(cfg)
    for _ in range(30): env.step([])

    ########## Initiate camera ########################
    cam = VisionSensor(
            prim_path="/World/viewer_camera",
            name="camera",
            modalities=["rgb", "bbox_2d_loose", "bbox_2d_tight", "seg_semantic", "seg_instance"], #"depth_linear", "seg_instance", "bbox_2d_tight", "bbox_3d", "camera"],
            image_height=1024,
            image_width=1024,
        )
    cam.initialize()
    # Allow camera teleoperation
    og.sim.enable_viewer_camera_teleoperation()
    cam.set_position([3, 3, 5])

    cam_light = LightObject(prim_path="/World/mylight", 
                          name="mylight", 
                          light_type="Disk", 
                          radius=0.03, 
                          intensity=1e7)

    return env, cam, cam_light


def sample_cam_pose(yaw_low=-np.pi, yaw_high=np.pi, dist_low=2.5, dist_high=4.5, pitch_low=-np.pi/4, pitch_high=np.pi/16, obj=None):
    '''
    Helper function to saple a random camera pose.
    Later used to let cam focus on given object.
    
    obj: the DatasetObject to make camera focusing on. Note we still just return the camera pose.
    If passed in, will use the orientation of the obj to make sure camera is in the front side.
    '''
    if obj is None:
        camera_yaw = np.random.uniform(yaw_low, yaw_high)
        # camera_yaw = clipped_normal_sample(yaw_low, yaw_high)
    else:
        camera_yaw = compute_obj_x_pos_angle(obj) # let camera face obj for debugging
        camera_yaw = np.random.uniform(camera_yaw - np.pi/8, camera_yaw + np.pi/8)
    
    camera_dist = np.random.uniform(dist_low, dist_high)
    camera_pitch = np.random.uniform(low=pitch_low, high=pitch_high)
    # camera_pitch = clipped_normal_sample(pitch_low, pitch_high)
    target_to_camera = R.from_euler("yz", [camera_pitch, camera_yaw]).apply([1, 0, 0])
    raw_camera_pos = target_to_camera * camera_dist

    return (raw_camera_pos, camera_yaw)

def compute_obj_x_pos_angle(obj):
    orn = obj.get_orientation()
    mat = T.quat2mat(orn)
    forward_vec = -mat[:, 1]
    return np.pi * 2 - compute_angle_between_vectors(forward_vec, np.array([1, 0, 0]))


def compute_angle_between_vectors(vector_a, vector_b):
    cos_angle = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle

def clipped_normal_sample(low, high):
    '''
    Samples from a normal distribution centered at (low+high)/2
    std = abs(high-low)/6
    clipped to [low, high]
    '''
    return np.clip(np.random.normal((low+high)/2, np.abs(high-low)/6, 1)[0], low, high)

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
        category="top_cabinet",
        model="dmwxyl",
    )

    cab1_name = f"cab_{1}"
    cab1 = DatasetObject(
        prim_path=f"/World/{cab1_name}",
        name=cab1_name,
        category="bottom_cabinet",
        model="bamfsz",
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
        model="bmnubh"
    )

    shirt = DatasetObject(
        prim_path="/World/shirt",
        name="shirt",
        category="t_shirt", 
        model="kvidcx", 
        prim_type=PrimType.CLOTH, 
        abilities={"foldable": {}, "unfoldable": {}},
        # scale=0.05,
        fit_avg_dim_volume=True,
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

    apple = DatasetObject(
        prim_path="/World/apple",
        name="apple",
        category="apple", 
        model="agveuv", 
        scale=1.5,
    )

    knife = DatasetObject(
        prim_path="/World/knife",
        name="knife",
        category="table_knife",
        model="lrdmpf",
        scale=2.5,
        position=[0, 0, 10.0],
    )

    return cab0, cab1, cupcake, laptop, table, shirt, carpet, apple, knife


if __name__ == '__main__':
    print("here")
    embed()
    