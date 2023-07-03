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
import copy

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
from omnigibson.utils.usd_utils import SemanticsAPI

# Utility imports
from IPython import embed
import matplotlib.pyplot as plt
from omni.isaac.synthetic_utils.visualization import colorize_bboxes
import json
from omnigibson.utils.collectData.openable_obj import (
    get_objects_by_categories,
    get_scene_objects_by_categories,
    get_all_place_objects,
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
    create_predefined_env,
)
from omnigibson.utils.collectData.obs_utils import (
    save_link_level_obs,
    get_bounding_box,
)

# gm global vars
# gm.USE_GPU_DYNAMICS = False
# gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True

# Collect pipeline global vars
OPENABLE_CATEGORIES = ["bottom_cabinet", "top_cabinet", "bottom_cabinet_no_top"]
UNDER_CATEGORIES = ["breakfast_table", "coffee_table", "console_table", "desk", "pedestal_table", "straight_chair"]
ONTOP_CATEGORIES = ['armchair', "countertop","breakfast_table", "coffee_table", "console_table", "desk", "pedestal_table", "shelf", "straight_chair", "swivel_chair"]
ORIENTED_CATEGORIES = ["bottom_cabinet", "top_cabinet", "bottom_cabinet_no_top", 'armchair', "shelf"]

def randomize_obj_state(base_objects, place_objects, small_categories, 
                        one_link=False, opened_links=None, place_state=None, sample_inside_link=False):
    '''
    Randomly samples one state to place for each place_obj
    instance_map: instance mapping obtained by sem_api
    resample_base: if not resample, provide opened_links
    resample_place: if not resample, provide place_state
    sample_inside_link: for PARISMATIC drawers, pass this argument to True in order to sample object
    only inside opened link
    Return: all_success: boolean. state_obs:dict
    '''
    all_success = True
    state_obs = {"predicates":[]}

    # Currently assumes only one base_object
    # Need to make place_states a dict if adding more
    if opened_links is None:
        for base_obj in base_objects:
            if base_obj.category in OPENABLE_CATEGORIES: 
                is_open = np.random.choice([True, False], 1, p=[0.9, 0.1])[0] if one_link is False else True
                base_obj.states[Open].set_value(False) # first close the cabinet anyways
                open_success, links = base_obj.states[Open].set_value(is_open, fully=False, one_link=one_link)
                for _ in range(10): og.sim.step()

                all_success = all_success and open_success
                if (links is not None and is_open):
                    inside_sample_link_name = '_'.join(np.random.choice(links).name.split('_')[-2:]) # convert joint name to link name
                else:
                    inside_sample_link_name = None # only put object inside one of opened links
                
                opened_links = ''.join([link.name for link in links]) if is_open else ''
    
    if not all_success:
        og.log.info("Opening cabinet failed, returning")
        return False, None, ""
    
    place_obj = place_objects[0]
    if place_state is None:
        place_states = []
        place_state = OnTop
        if base_obj.category in OPENABLE_CATEGORIES: place_states.append(Inside)
        if base_obj.category in ONTOP_CATEGORIES: place_states.append(OnTop)
        if base_obj.category in UNDER_CATEGORIES: place_states.append(Under)
        if len(place_states) == 0: return False, None, ""

        for place_obj in place_objects:
            cur_place_states = copy.deepcopy(place_states)
            if place_obj.category not in small_categories:
                # if OnTop not in place_states: continue
                # cur_place_states = [OnTop]
                if Inside in cur_place_states:
                    if len(cur_place_states) == 1: continue
                    else: cur_place_states.remove(Inside)
            place_state = np.random.choice(cur_place_states, 1)[0]
            if place_state == Inside and sample_inside_link:
                place_success = place_obj.states[Inside].set_value(base_obj, True, sample_link_name=inside_sample_link_name)
            else:
                place_success = place_obj.states[place_state].set_value(base_obj, True)
            og.log.info(f"Placed {place_obj.category} {place_state} {base_obj.category}, sucess {place_success}")
            all_success = all_success and place_success

    for _ in range(15): og.sim.step()
    place_success = place_obj.states[place_state].get_value(base_obj)
    if not place_success:
        all_success = False
        og.log.info("Object state False after env step")

    if all_success: # process state observation
        # store place state
        base_obj_name = base_obj.name.split("_")
        base_obj_name = f"{base_obj_name[-2]}_{base_obj_name[-1]}"
        place_obj_name = place_obj.name.split("_")
        place_obj_name = f"{place_obj_name[-2]}_{place_obj_name[-1]}"
        edge_name = (base_obj_name, place_obj_name)
        edge_type = ["OnTop", "Inside", "Under"][[OnTop, Inside, Under].index(place_state)]
        state_obs["predicates"].append([edge_name[0], edge_name[1], edge_type])

    return all_success, state_obs, opened_links, place_state


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


def save_obj_obs(base_obj, place_obj, opened_links, instance_map, prefix, fpath, cam):
    '''
    Also check if visible.
    '''
    obj_obs = {}
    all_visible = True
    open_visible = False
    for obj in [base_obj] + [place_obj]:
        obj_id = instance_map[f"/World/{obj.name}"]
        if obj in [place_obj]:
            obj_id += 1
        obj_name = obj.name.split("_")
        obj_name = f"{obj_name[-2]}_{obj_name[-1]}"

        obj_obs[obj_name] = {
            "category":obj.category,
            "is_articulated": obj.category in OPENABLE_CATEGORIES,
            "links": None,
            "bbox": get_bounding_box(cam.get_obs()["seg_instance"], obj_id)
        }

        if obj.category in OPENABLE_CATEGORIES:
            link_obs = save_link_level_obs(obj, obj_id, opened_links, prefix=prefix, fpath=fpath, cam=cam)
            if link_obs is None:
                all_visible = False
                open_visible = False
            else:
                open_visible = True
            obj_obs[obj_name]["links"] = link_obs
        else:
            # TODO: check inserted object's obj_id, and write filtering method here
            if np.sum(cam.get_obs()["seg_instance"]==obj_id) < 300:
                # embed()
                all_visible = False
    
    return obj_obs, all_visible, open_visible


def dump_json(info, fpath):
    with open(f"{fpath}info.json", 'w') as f:
        json.dump(info, f, indent=4)
    f.close()

######################## MAIN GENERATION FUNCTION ######################
def generate(scene_id=0, base_categories=["bottom_cabinet"], is_train=True, fpath="collected_data/pipeline_test/"):
    
    ########### Create Environment and Viewer Camera #########
    # env, cam = create_env_with_light()
    env, cam, cam_light = create_predefined_env(scene_id, load_categories=base_categories+['countertop',"fridge", "microwave", "picture"])
    og.sim.stop()
    og.sim.import_object(cam_light)
    og.sim.play()
    sem_api = SemanticsAPI()

    ########### Get all articulate objects ############
    place_objects, small_categories = get_all_place_objects(is_train, unique_id="0")
    # place_objects = get_objects_by_categories(place_categories, use_avg_spec=True, is_train=is_train, unique_id="0")
    # base_objects = get_scene_objects_by_categories(base_categories)
    # base_objects = get_scene_objects_by_categories(OPENABLE_CATEGORIES)
    base_objects = get_scene_objects_by_categories(["bottom_cabinet", "bottom_cabinet_no_top", "top_cabinet"])
    print(len(base_objects))

    ########### Prepare Generation #####################
    info = {}
    count = 0
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    if not os.path.exists(f"{fpath}/imgs/"):
        os.makedirs(f"{fpath}/imgs/")

    ########### Insert All Place Objects ###################
    # temporary solution to avoid segmentation id mismatch
    place_objects = place_objects
    for i in range(len(place_objects)):
        insert_success = insert_object(env, place_objects[i], position=[30+i, 0, 0])
        if not insert_success: 
            place_objects[i] = None # manually mask the uninserted objects

    ########### Iterate Over Objects to Generate Random States ##########
    for i in range(len(base_objects)):

        # re-creating place objects for re-insertion
        base_obj = base_objects[i]

        # skip drawers for now
        is_prismatic = False
        for joint_name in base_obj.joints:
            if "Prismatic" in base_obj.joints[joint_name].joint_type:
                is_prismatic = True
        if not is_prismatic: # TODO remove 0531
            og.log.info("not has prismatic joint, skipping for now")
            continue
        
        opened_links = None
        place_state = None
        all_success = False

        # insert place object
        for place_obj in np.random.choice(place_objects, 15, replace=False):
            
            # skip place objects that we failed to insert
            if not place_obj: continue

            # sample a few random states
            sample_itr = 4

            for itr in range(sample_itr):
                if is_prismatic:
                    all_success, state_obs, opened_links, place_state = randomize_obj_state(
                        base_objects=[base_obj], place_objects=[place_obj], small_categories=small_categories,
                        one_link=True, sample_inside_link=is_prismatic) # only specify inside link for drawers
                else:
                    all_success, state_obs, opened_links, place_state = randomize_obj_state(
                        base_objects=[base_obj], place_objects=[place_obj], small_categories=small_categories) 
                    
                if not all_success:
                    og.log.info(f"Sampling object states failed, skipped")
                    continue

                num_cam_pose = 4
                for _ in range(num_cam_pose):
                    ref_position = place_obj.get_position()
                    
                    if base_obj.get_position()[2] < 1.5:
                        # pitch_low, pitch_high = -np.pi/4, -np.pi/8
                        pitch_low, pitch_high = -np.pi*7/16, -np.pi/8 # 0531 for drawer inside
                        ref_position[2] += 0.2
                    else:
                        pitch_low, pitch_high = -np.pi/8, np.pi/16
                        
                    # sample camera pose -- currently always let camera focus on base obj
                    if base_obj.category in ORIENTED_CATEGORIES: 
                        cam_pos = sample_cam_pose(dist_low=1, dist_high=2, pitch_low=pitch_low, pitch_high=pitch_high, obj=base_obj)[0] + place_obj.get_position()
                    else:
                        cam_pos = sample_cam_pose(dist_low=1.5, dist_high=2.5, pitch_low=pitch_low, pitch_high=pitch_high)[0] + place_obj.get_position()
                    set_camera_view(cam_pos, ref_position, camera_prim_path="/World/viewer_camera", viewport_api=None)
                    for _ in range(30): og.sim.render()

                    # add camera light
                    cam_light.set_position_orientation(position=cam.get_position(), orientation=cam.get_orientation())
                    cam_light.intensity = np.random.uniform(5e5, 2e6)
                    for _ in range(50): og.sim.step()

                    if not place_obj.states[place_state].get_value(base_obj): continue

                    prefix = f"{count}"

                    obj_obs, all_visible, open_visible = save_obj_obs(base_obj, place_obj, opened_links, sem_api.get_instance_mapping(), prefix, fpath, cam)

                    if all_visible:
                        info[prefix] = {"objects":obj_obs,
                                        "predicates": state_obs["predicates"],}
                    # elif open_visible:
                    #     info[prefix] = {"objects":obj_obs}
                    else:
                        og.log.info("Not all objects visible, skipping")
                        continue 

                    rgb = cam.get_obs()["rgb"][:, :, :3]
                    seg = cam.get_obs()["seg_semantic"]

                    plt.imsave(f"{fpath}/imgs/{prefix}.png", rgb)
                    plt.imsave(f"{fpath}/imgs/{prefix}_seg.png", seg)

                    count += 1

            # og.sim.scene.remove_object(place_obj)
            place_obj.set_position([30+place_objects.index(place_obj), 0, 0])
            for _ in range(20): og.sim.step()

        # set opened base object to close
        if base_obj.category in OPENABLE_CATEGORIES:
            base_obj.states[Open].set_value(False)
            for _ in range(30): og.sim.step()
        # for _ in range(20): og.sim.render()
        dump_json(info, fpath=fpath)

    og.log.info("generation done, logging info to json")
    dump_json(info, fpath=fpath)

    env.close()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=int, default=0)
    args = parser.parse_args()

    generate(scene_id=args.scene_id, 
            # base_categories=list(set(OPENABLE_CATEGORIES+ONTOP_CATEGORIES+UNDER_CATEGORIES)), 
            base_categories=list(set(OPENABLE_CATEGORIES)), 
            fpath=f"collected_data/0531_Inside_Drawer_test/scene_{args.scene_id}/",
            is_train=False)
    # ['apple', 'bowl', 'peach','cup', 'hat']