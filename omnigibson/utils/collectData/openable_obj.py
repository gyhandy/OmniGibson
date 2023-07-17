import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_all_object_category_models,
)
from IPython import embed
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.synthetic_utils.visualization import colorize_bboxes
import matplotlib.pyplot as plt
from omnigibson.object_states import Open
import random

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True


# global variables 
NOT_USE_AVG_SPEC_CATEGORIES = ["bottom_cabinet"]
SKIP_MODELS = {"bottom_cabinet":["cjcyed", "gvtucm", "leizjb", "olgoza", "phoesw", "vespxk"]}


def get_all_available_by_category(category, is_train, use_avg_spec=None, unique_id=None, abilities={}):
    '''
    Returns (train, test), two list of DatasetObjects for all 
    available models in the specified category.
    Notice: Does not handle invalid articulations
    use_avg_spec: if not pass in, will be set True unless category in NOT_USE_AVG_SPEC_CATEGORIES
    '''
    all_models = get_all_object_category_models(category)
    train = []
    test = []
    
    if use_avg_spec is None:
        use_avg_spec = category not in NOT_USE_AVG_SPEC_CATEGORIES
        print(category, use_avg_spec)

    for i in range(len(all_models)): 
        if category in SKIP_MODELS and all_models[i] in SKIP_MODELS[category]:
            continue
        if unique_id is None:
            name = f"{all_models[i]}"
        else:
            name = f"{all_models[i]}_{unique_id}" # set name = model
        obj = DatasetObject(
            prim_path=f"/World/{name}",
            name=name,
            category=category,
            model=all_models[i],
            fit_avg_dim_volume=use_avg_spec,
            abilities=abilities,
        )
        if len(all_models) == 1:
            if is_train: train.append(obj)
            else: test.append(obj)
            return train, test
        
        elif i < len(all_models) * 0.6:
            train.append(obj)
        else:
            test.append(obj)

    return train, test


def get_objects_by_categories(categories, use_avg_spec=None, is_train=True, unique_id=""):
    train = []
    test = []
    for category in categories:
        cur_train, cur_test = get_all_available_by_category(category, is_train, use_avg_spec, unique_id )
        train += cur_train
        test += cur_test
        if (is_train and len(cur_train) > 30) or (not is_train and len(cur_test) > 30):
            break
        og.log.info(f"Collected category {category}, {len(cur_train)} training, {len(cur_test)} testing objects")

    if is_train:
        return train
    else:
        return test
    
def get_scene_objects_by_categories(categories):
    '''
    Obtains all objects in predefined scene in specified categories
    '''
    objects = []
    for category in categories:
        cur_cat = og.sim.scene.object_registry("category", category)
        objects += list(cur_cat) if cur_cat is not None else []

    return objects

def get_all_place_objects(is_train=True, unique_id = ""):
    small_place_categories = []
    place_categories = []

    for cat in get_all_object_categories():
        metadata = get_og_avg_category_specs().get(cat)
        if metadata is None or metadata["size"] is None or metadata["mass"] is None:
            print(cat)
            continue
        size = max(metadata["size"])
        weight = metadata['mass']

        if size < 0.5 and size > 0.1 and weight < 10:
            place_categories.append(cat)
        if size < 0.3 and size > 0.1:
            small_place_categories.append(cat)

    train_place_categories = place_categories[:int(0.7*len(place_categories))]
    test_place_categories = place_categories[int(0.7*len(place_categories)):]
    if is_train:
        random.shuffle(train_place_categories)
        place_objects = get_objects_by_categories(train_place_categories, True, is_train, unique_id)
    else:
        random.shuffle(test_place_categories)
        place_objects = get_objects_by_categories(test_place_categories, True, is_train, unique_id)
    random.shuffle(place_objects)
    place_objects = place_objects[:7]

    return place_objects, small_place_categories

def show_obs(obs):
    rgb = obs["rgb"][:, :, :3]
    rgb_instance = obs['seg_instance'][0]
    rgb_with_bbox = colorize_bboxes(bboxes_2d_data=obs["bbox_2d_loose"], bboxes_2d_rgb=obs["rgb"], num_channels=4)

    plt.imshow(rgb_instance)
    plt.show()

    plt.imshow(rgb_with_bbox)
    plt.show()


if __name__ == '__main__':
    # train, test = get_all_available_by_category("bottom_cabinet")
    embed()
    exit(0)

    ########### Create Empty Environment ##############
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
            modalities=["rgb", "seg_instance","bbox_2d_loose", "bbox_2d_tight"], #"depth_linear", "seg_instance", "bbox_2d_tight", "bbox_3d", "camera"],
            image_height=1024,
            image_width=1024,
        )
    cam.initialize()
    # Allow camera teleoperation
    og.sim.enable_viewer_camera_teleoperation()

    cab_name = f"cab_{0}"
    cab = DatasetObject(
        prim_path=f"/World/{cab_name}",
        name=cab_name,
        category="bottom_cabinet",
        model="qacthv"
    )

    embed()
    env.close()
