'''
Script to generate dataset of objects with binary relations
Currently support: OnTop, Inside, Under, Open/Close
TODO: more detailed documentation on the generation process
'''


# Functionality imports
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import OrderedDict
import os

# OmniGibson imports
import omnigibson as og
from omnigibson.scenes import Scene
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
from omnigibson.object_states import Open, OnTop, Inside, Under, Folded, Unfolded, Overlaid
import omnigibson.utils.transform_utils as T

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
    plot_all_bboxes,
    new_get_all_bboxes,
    xy_minmax_to_center_extend_2d,
)
from omnigibson.utils.collectData.env_utils import (
    create_env_with_light,
    sample_cam_pose,
)

# gm global vars
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True

# Collect pipeline global vars
OPENABLE_CATEGORIES = ["bottom_cabinet"]
TABLE_CATEGORIES = ["breakfast_table", "coffee_table"]
ONTOP_ONLY_CATEGORIES = ['pot_plant']


def save_link_level_obs(obj, opened_links, prefix, info, fpath="collected_data/small_sample/", cam=None):
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
        return info
    
    # TODO: 
    link_bboxes, bbox_img = get_all_bboxes(obj, cam)

    rgb = obs["rgb"][:, :, :3]
    rgb_with_bbox = bbox_img

    plt.imsave(f"{fpath}/imgs/{prefix}.png", rgb)
    plt.imsave(f"{fpath}/imgs/{prefix}_bbox.png", rgb_with_bbox)
    
    # process opened link here
    info[prefix] = {}
    for key in link_bboxes:
        info[prefix][key] = {}
        info[prefix][key]["bbox"] = link_bboxes[key] # bbox in corner-extent format
        if key in ["object", "base_link"]:
            is_open = None
        else:
            is_open = key in opened_links
        info[prefix][key]["is_open"] = is_open
    
    return info

def randomize_obj_state(base_objects, place_objects):
    '''
    Randomly samples one state to place for each place_obj
    Return: all_success: boolean. state_obs:dict
    '''
    all_success = True
    state_obs = {"objects":{},
                 "predicates":[]}

    # store object_level observations
    for obj in base_objects + place_objects:
        state_obs["objects"][obj.name.split("_")[0]] = {
            "category":obj.category,
            "is_articulated": obj.category in OPENABLE_CATEGORIES,
            "links": list(obj.links.keys()) if obj.category in OPENABLE_CATEGORIES else None,
        }

    # Currently assumes only one base_object
    # Need to make place_states a dict if adding more
    for base_obj in base_objects:
        if base_obj.category in OPENABLE_CATEGORIES: 
            is_open = np.random.choice([True, False], 1, p=[0.8, 0.2])[0]
            base_obj.states[Open].set_value(False) # first close the cabinet anyways
            open_success, links = base_obj.states[Open].set_value(is_open, fully=True)
            for _ in range(10): og.sim.step()

            all_success = all_success and open_success
            opened_links_str = ''.join([link.name for link in links]) if is_open else ''
            place_states = [OnTop, Inside] if is_open else [OnTop]
            # state_obs[...] = ...
        else:
            place_states = [OnTop, Under]
    
    if not all_success:
        return False, None

    for place_obj in place_objects:
        if place_obj.category in ONTOP_ONLY_CATEGORIES:
            place_states = [OnTop]
        place_state = np.random.choice(place_states, 1)[0]
        place_success = place_obj.states[place_state].set_value(base_obj, True)
        og.log.info(f"Placed {place_obj.category} {place_state} {base_obj.category}, sucess {place_success}")
        all_success = place_success and place_success
        # state_obs[...] = ...

    for _ in range(15): og.sim.step()
    place_success = place_obj.states[place_state].get_value(base_obj)
    if not place_success:
        all_success = False
        og.log.info("Object state False after env step")

    if all_success: # process state observation

        # store open/close
        if base_obj.category in OPENABLE_CATEGORIES:
            for link_name in list(base_obj.links.keys()):
                if link_name == "base_link":
                    continue
                base_obj_name = base_obj.name.split("_")[0]
                edge_name = (f"{base_obj_name}-base_link", f"{base_obj_name}-{link_name}")
                edge_type = "Open" if link_name in opened_links_str else "Close"
                state_obs["predicates"].append([edge_name[0], edge_name[1], edge_type])

        # store place state
        edge_name = (base_obj.name.split("_")[0], place_obj.name.split("_")[0])
        edge_type = ["OnTop", "Inside", "Under"][[OnTop, Inside, Under].index(place_state)]
        state_obs["predicates"].append([edge_name[0], edge_name[1], edge_type])

    return all_success, state_obs


def check_visible(cam, objects=None):
    '''
    Checks if all objects of interest are visible with camera segmentation map
    Return: bool
    '''
    all_visible = True

    obs_instance = cam.get_obs()["seg_instance"]
    seg_map = obs_instance[0]
    metadata = obs_instance[1]

    # currently just hack, in future select object of interest by category
    # obj_ids = range(1, len(metadata)+1)  
    obj_ids = []
    for data in metadata:
        obj_ids.append(list(data)[0])
    for obj_id in obj_ids:
        relevant = np.sum(seg_map == obj_id)
        all_visible = all_visible and (relevant > 400)

    # embed()

    return all_visible


def insert_object(env, obj, position=[0., 0., 0.15]):
    '''
    Helper function to insert some object into env
    Return: bool -- insertion success
    '''
    insert_success = None
    try:
        # insert obj into scene
        og.sim.stop()
        og.sim.import_object(obj)
        obj.set_position(position)
        og.sim.play()
        # set right after insertion to avoid physical simulation issue
        # TODO: ask Eric if insertion causes collision with other object is fine
        # obj.set_position([0, 0, 0.15]) 
        for _ in range(50): env.step([])
        for _ in range(30): og.sim.render()
        insert_success = True

    # some models may raise error for invalid articulation handle
    except Exception as e:
        og.log.info(f"AssertionError encountered, failed to import object {obj.name}")
        og.log.info(f"{str(e)}")
        og.sim.scene.remove_object(obj)
        embed()
        insert_success = False

    return insert_success


def get_bbox_obs(cam, base_obj, place_obj):
    '''
    NOTE: currently hard-coded that there are two objects, first one is 
    base_obj and second one is place_obj
    '''
    orig_obs = cam.get_obs()["bbox_2d_loose"]

    if base_obj.category in OPENABLE_CATEGORIES:
        base_link_bboxes, bbox_img = new_get_all_bboxes(base_obj, cam)
        bboxes = base_link_bboxes
        bboxes[place_obj.name.split("_")[0]] = xy_minmax_to_center_extend_2d(list(orig_obs[-1])[-4:])

    else:
        bboxes = {}
        bboxes[place_obj.name.split("_")[0]] = xy_minmax_to_center_extend_2d(list(orig_obs[-1])[-4:])
        bboxes[base_obj.name.split("_")[0]] = xy_minmax_to_center_extend_2d(list(orig_obs[0])[-4:])
        bbox_img = colorize_bboxes(bboxes_2d_data=orig_obs, bboxes_2d_rgb=cam.get_obs()["rgb"], num_channels=4)
    
    return bboxes, bbox_img

def dump_json(info, fpath):
    with open(f"{fpath}info.json", 'w') as f:
        json.dump(info, f, indent=4)
    f.close()

######################## MAIN GENERATION FUNCTION ######################
def generate(base_categories=["bottom_cabinet"], place_categories=["apple"], is_train=True, fpath="collected_data/pipeline_test/"):
    
    ########### Create Environment and Viewer Camera #########
    env, cam = create_env_with_light()

    ########### Get all articulate objects ############
    base_objects = get_objects_by_categories(base_categories, is_train=is_train, unique_id="0")
    # place_objects = get_objects_by_categories(place_categories, use_avg_spec=True, is_train=is_train)

    ########### Prepare Generation #####################
    info = {}
    count = 0
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    if not os.path.exists(f"{fpath}/imgs/"):
        os.makedirs(f"{fpath}/imgs/")

    ########### Iterate Over Objects to Generate Random States ##########
    for i in range(len(base_objects)):

        # re-creating place objects for re-insertion
        base_obj = base_objects[i]
        place_objects = get_objects_by_categories(place_categories, is_train=is_train, unique_id=str(i))

        # insert base_obj into env
        insert_success = insert_object(env, base_obj)
        if not insert_success:
            og.log.info(f"Import failed for object {base_obj.name}, skipping")
            continue
        og.log.info(f"Successfully imported object {base_obj.name}")

        # check if have original bbox
        if len(cam.get_obs()['bbox_2d_loose']) == 0:
            og.log.info(f"object {base_obj.name} has no original bboxes, skipped")
            og.sim.scene.remove_object(base_obj)
            continue

        # insert place object
        # for place_obj in np.random.choice(place_objects, 30, replace=False):
        for place_obj in place_objects:
            insert_success = insert_object(env, place_obj, position=[3, 3, 0.15])
            if not insert_success:
                og.log.info(f"Import failed for object {place_obj.name}, skipping")
                continue
            og.log.info(f"Successfully imported object {place_obj.name}")

            # sample a few random states
            sample_itr = 3

            for itr in range(sample_itr):
                # TODO: sample a state for the objects (e.g. open, placement, etc.)
                all_success, state_obs = randomize_obj_state(base_objects=[base_obj], place_objects=[place_obj])

                if not all_success:
                    og.log.info("Sampling object states failed, skipped")
                    continue

                num_cam_pose = 5
                for _ in range(num_cam_pose):
                    # sample camera pose -- currently always let camera focus on base obj
                    cam_pos = sample_cam_pose(dist_high=4) + place_obj.get_position()
                    set_camera_view(cam_pos, place_obj.get_position(), camera_prim_path="/World/viewer_camera", viewport_api=None)
                    for _ in range(30): og.sim.render()

                    # check if all objects visible under this camera pose
                    all_visible = check_visible(cam)
                    og.log.info(f"All visible status {all_visible}")
                    if not all_visible: continue

                    # TODO: define prefix in a better manner
                    prefix = f"{count}"
                    
                    # save bbox and rgb observation here
                    bbox_obs, bbox_img = get_bbox_obs(cam, base_obj, place_obj)
                    info[prefix] = {"objects":state_obs["objects"],
                                    "predicates": state_obs["predicates"],
                                    "bboxes": bbox_obs}
                    rgb = cam.get_obs()["rgb"][:, :, :3]
                    rgb_with_bbox = bbox_img

                    plt.imsave(f"{fpath}/imgs/{prefix}.png", rgb)
                    plt.imsave(f"{fpath}/imgs/{prefix}_bbox.png", rgb_with_bbox)

                    # embed()

                    count += 1

            og.sim.scene.remove_object(place_obj)
            for _ in range(20): og.sim.render()

        # remove object
        og.sim.scene.remove_object(base_obj)
        for _ in range(20): og.sim.render()
        dump_json(info, fpath=fpath)

    og.log.info("generation done, logging info to json")
    dump_json(info, fpath=fpath)

    env.close()

if __name__ == '__main__':

    # generate(categories=['bottom_cabinet'], fpath="collected_data/0406_bottom_cab_link_level_test/")
    generate(base_categories=["breakfast_table"], 
             place_categories=['pot_plant', 'table_knife', 'toy', 'apple', 'bowl', 'peach','cup', 'hat', 'jar'],
             fpath="collected_data/0412_breakfast_table_binary_test/",
             is_train=False)
    # ['apple', 'bowl', 'peach','cup', 'hat']