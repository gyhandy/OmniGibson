'''
Script to generate dataset of open/closed objects
Gets one cabinet/fridge/microwave at once, generate open states/camera pose
for 5 times, move on to the next.
'''


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
from omnigibson.utils.usd_utils import SemanticsAPI

# Utility imports
from IPython import embed
import matplotlib.pyplot as plt
from omni.isaac.synthetic_utils.visualization import colorize_bboxes
import json
from omnigibson.utils.collectData.openable_obj import (
    get_objects_by_categories,
)
from omnigibson.utils.collectData.bbox_utils import (
    get_all_bboxes,
    plot_bbox_on_rgb
)
from omnigibson.utils.collectData.env_utils import (
    create_env_with_light,
    create_predefined_env,
    sample_cam_pose,
)
from omnigibson.utils.collectData.filter_utils import (
    check_visible
)

import argparse

# gm global vars
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True

# def sample_cam_pose(yaw_low=-np.pi, yaw_high=0, dist_low=2.5, dist_high=4.5, pitch_low=-np.pi/4, pitch_high=np.pi/16):
#     '''
#     Helper function to saple a random camera pose.
#     Later used to let cam focus on given object.
#     '''
#     camera_yaw = np.random.uniform(yaw_low, yaw_high)
#     camera_dist = np.random.uniform(dist_low, dist_high)
#     camera_pitch = np.random.uniform(low=pitch_low, high=pitch_high)
#     target_to_camera = R.from_euler("yz", [camera_pitch, camera_yaw]).apply([1, 0, 0])
#     raw_camera_pos = target_to_camera * camera_dist

#     return raw_camera_pos

def save_obs(prefix, is_open, info, fpath="collected_data/small_sample/", cam=None):
    '''
    Helper function to save current obs to fig
    '''
    if not cam:
        cam = og.sim.viewer_camera
    
    obs = cam.get_obs()

    if len(obs["bbox_2d_tight"]) == 0:
        # not visible, don't save
        return info

    rgb = obs["rgb"][:, :, :3]
    rgb_instance = obs['seg_instance'][0]
    rgb_with_bbox = colorize_bboxes(bboxes_2d_data=obs["bbox_2d_loose"], bboxes_2d_rgb=obs["rgb"], num_channels=4)
    
    bbox_loose = list(map(float, list(obs["bbox_2d_loose"][0])[-4:]))
    left_corner_loose = (bbox_loose[0], bbox_loose[1])
    span_loose = (bbox_loose[2]-bbox_loose[0], bbox_loose[3]-bbox_loose[1])

    plt.imsave(f"{fpath}/imgs/{prefix}.png", rgb)
    plt.imsave(f"{fpath}/imgs/{prefix}_bbox.png", rgb_with_bbox)
    plt.imsave(f"{fpath}/imgs/{prefix}_seg_ins.png", rgb_instance)
    # info[prefix] = {"bbox_loose":(left_corner_loose, span_loose), "bbox_tight":(left_corner_tight, span_tight), "is_open":is_open}
    info[prefix] = {"bbox_loose":(left_corner_loose, span_loose), "is_open":is_open}
    return info

def save_link_level_obs(obj, obj_id, opened_links, prefix, info, fpath="collected_data/small_sample/", cam=None):
    '''
    Helper function to save current obs to fig
    obj: DatasetObject -- TODO: extend for multiple objects
    opened_links: string of all opened links concatenated
    prefix: data point name
    info: previous info dictionary
    fpath: directory to save images
    cam: camera object to capture observations
    '''
    if not cam:
        # cam = og.sim.viewer_camera
        og.log.info("Please pass in cam")
        return info
    
    obs = cam.get_obs()

    if len(obs["bbox_2d_tight"]) == 0:
        # not visible, don't save
        og.log.info("no bbox, skipping")
        return info
    
    # TODO: select only wanted object in bbox
    raw_bbox_obs = cam._annotators["bbox_2d_loose"].get_data()
    bbox_obs = [raw_bbox_obs['data'][i] for i in range(len(raw_bbox_obs['data'])) if f"/World/{obj.name}" in raw_bbox_obs["info"]["primPaths"][i]]
    
    # bbox_obs = obs["bbox_2d_loose"]
    # bbox_obs = [ob for ob in bbox_obs if f"/World/{obj.name}" in ob[1]]
    link_bboxes, bbox_obs = get_all_bboxes(obj, cam, bbox_obs=bbox_obs, img_fpath=f"{fpath}/imgs/{prefix}_bbox.png")

    # TODO: check visibility here
    min_ratio = check_visible(obs["seg_instance"], obj_id, bbox_obs)
    print(prefix, min_ratio)
    if min_ratio < 0.2:
        og.log.info("Not all links properly visible, skipping")
        return info

    rgb = obs["rgb"][:, :, :3]
    plt.imsave(f"{fpath}/imgs/{prefix}.png", rgb)
    # plt.imsave(f"{fpath}/imgs/{prefix}_seg.png", obs["seg_instance"]==obj_id)
    plot_bbox_on_rgb(bboxes=bbox_obs, rgb=rgb, fpath=f"{fpath}/imgs/{prefix}_bbox.png", is_minmax=True)
    # plt.imsave(f"{fpath}/imgs/{prefix}_bbox.png", bbox_img)

    # embed()

    # mask = obs['seg_semantic'] == 0
    # mask_rgb = np.stack([mask, mask, mask], axis=-1)
    # rgb_white_bg = rgb
    # rgb_white_bg[mask_rgb] = 255.

    # plt.imsave(f"{fpath}/imgs/{prefix}_white_bg.png", rgb_white_bg)
    
    # process opened link here
    info[prefix] = {}
    info[prefix]["min_link_visible_ratio"] = min_ratio
    for key in link_bboxes:
        info[prefix][key] = {}
        info[prefix][key]["bbox"] = link_bboxes[key] # bbox in corner-extent format
        if key in ["object", "base_link"]:
            is_open = None
        else:
            is_open = key in opened_links
        info[prefix][key]["is_open"] = is_open
    
    return info

def dump_json(info, fpath="collected_data/small_sample/"):
    with open(f"{fpath}info.json", 'w') as f:
        json.dump(info, f, indent=4)
    f.close()

######################## MAIN GENERATION FUNCTION ######################
def generate(scene_id=0, fpath="collected_data/pipeline_test/"):

    # env, cam = create_env_with_light()
    env, cam = create_predefined_env(scene_id)
    sem_api = SemanticsAPI()
    # embed()

    ########### Get all articulate objects ############
    
    cab_objects = og.sim.scene.object_registry("category", "top_cabinet")
    train_objects = list(cab_objects) if cab_objects is not None else []
    cab_objects = og.sim.scene.object_registry("category", "bottom_cabinet")
    train_objects += list(cab_objects) if cab_objects is not None else []

    ceiling_objects = og.sim.scene.object_registry("category", "ceilings")
    ceiling_objects = list(ceiling_objects) if ceiling_objects is not None else []
    for ceiling in ceiling_objects:
        og.sim.scene.remove_object(ceiling)
    for _ in range(50): og.sim.step()

    instance_map = sem_api.get_instance_mapping()

    # embed()
    # exit(0)

    info = {}
    count = 0

    if not os.path.exists(fpath):
        os.makedirs(fpath)
    if not os.path.exists(f"{fpath}/imgs/"):
        os.makedirs(f"{fpath}/imgs/")

    # iterate through available objects
    for obj in train_objects:
        obj_id = instance_map[f"/World/{obj.name}"]

        # 0421 Temporarily skip all prismatic joints
        has_prismatic_joint = False
        has_revolute_joint = False
        for joint_name in obj.joints:
            if "Prismatic" in obj.joints[joint_name].joint_type:
                has_prismatic_joint = True
            else:
                has_revolute_joint = True

        if has_prismatic_joint:
            og.log.info(f"object {obj.name} has prismatic joints, skipping")
            continue
        
        # itr_open = 6
        # itr_close = 3
        itr_open = 1
        itr_close = 1

        # keep track of visible ratio to find best angle for object
        best_yaw = -np.pi
        best_visible = 0

        # repeate several samples
        for itr in range(itr_open + itr_close):
            print(obj.name, itr, itr_open)
            if itr < itr_open: 
                obj.states[Open].set_value(False)
                success, links = obj.states[Open].set_value(True, fully=False)
                og.log.info(f"finished articulation, Opening {success}, Opened {len(links)} links")
                if not obj.states[Open].get_value():
                    og.log.info(f"Artibulation check failed, continue")
                    continue
                num_open = len(links)

            else:
                success, _ = obj.states[Open].set_value(False)
                og.log.info(f"finished articulation, Closing {success}")
                num_open = 0

            if not success:
                continue

            for _ in range(50): env.step([])

            num_cam_pose = 22
            yaws = np.linspace(-np.pi, np.pi, 19)
            
            for p in range(num_cam_pose):
                # rotate around object
                if p < 18 and itr == 0:
                    cam_pos, cam_yaw = sample_cam_pose(dist_low=1.5, dist_high=1.5, yaw_low=yaws[p], yaw_high=yaws[p+1])
                    cam_pos += obj.get_position()
                else:
                    cam_pos, _ = sample_cam_pose(dist_low=1, dist_high=3, yaw_low=best_yaw-np.pi/6, yaw_high=best_yaw+np.pi/6)
                    cam_pos += obj.get_position()

                set_camera_view(cam_pos, obj.get_position(), camera_prim_path="/World/viewer_camera", viewport_api=None)
                for _ in range(30): og.sim.render()

                # process the joint and get bbox for each opened link
                prefix = f"{count}"

                if num_open > 0:
                    opened_links = ''.join([link.name for link in links])
                else:
                    opened_links = ''
                info = save_link_level_obs(obj, obj_id, opened_links, prefix, info, fpath, cam)

                # help find the best camera yaw 
                if p < 18:
                    if prefix in info and best_visible - 0.05 < info[prefix]["min_link_visible_ratio"]:
                        best_visible = info[prefix]["min_link_visible_ratio"]
                        best_yaw = cam_yaw

                count += 1
            
            dump_json(info, fpath=fpath)

    og.log.info("generation done, logging info to json")
    dump_json(info, fpath=fpath)

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=int, default=0)
    args = parser.parse_args()

    generate(scene_id = args.scene_id,
             fpath=f"collected_data/0421_rev_only_train/scene_{args.scene_id}/"
             )