import omnigibson as og
from omnigibson.objects import DatasetObject
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_object_models_of_category,
)
from IPython import embed
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.sensors import VisionSensor
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.synthetic_utils.visualization import colorize_bboxes
import matplotlib.pyplot as plt
from omnigibson.object_states import Open

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True


# global variables 
NOT_USE_AVG_SPEC_CATEGORIES = ["bottom_cabinet"]
SKIP_MODELS = {"bottom_cabinet":["cjcyed", "gvtucm", "leizjb", "olgoza", "phoesw", "vespxk"]}


def get_all_available_by_category(category, use_avg_spec=None, unique_id=""):
    '''
    Returns (train, test), two list of DatasetObjects for all 
    available models in the specified category.
    Notice: Does not handle invalid articulations
    use_avg_spec: if not pass in, will be set True unless category in NOT_USE_AVG_SPEC_CATEGORIES
    '''
    all_models = get_object_models_of_category(category)
    train = []
    test = []
    if all_models == 1:
        og.log.info(f"Only one model available for {category}, skipping!")
        return train, test
    
    if use_avg_spec is None:
        use_avg_spec = category not in NOT_USE_AVG_SPEC_CATEGORIES
        print(category, use_avg_spec)

    for i in range(len(all_models)): 
        if category in SKIP_MODELS and all_models[i] in SKIP_MODELS[category]:
            continue
        name = f"{all_models[i]}_{unique_id}" # set name = model
        obj = DatasetObject(
            prim_path=f"/World/{name}",
            name=name,
            category=category,
            model=all_models[i],
            fit_avg_dim_volume=use_avg_spec,
        )
        if i < len(all_models) * 0.6 and i != len(all_models)-1:
            train.append(obj)
        else:
            test.append(obj)

    return train, test


def get_objects_by_categories(categories, use_avg_spec=None, is_train=True, unique_id=""):
    train = []
    test = []
    for category in categories:
        cur_train, cur_test = get_all_available_by_category(category, use_avg_spec, unique_id )
        train += cur_train
        test += cur_test
        og.log.info(f"Collected category {category}, {len(cur_train)} training, {len(cur_test)} testing objects")

    if is_train:
        return train
    else:
        return test

def show_obs(obs):
    rgb = obs["rgb"][:, :, :3]
    rgb_instance = obs['seg_instance'][0]
    rgb_with_bbox = colorize_bboxes(bboxes_2d_data=obs["bbox_2d_loose"], bboxes_2d_rgb=obs["rgb"], num_channels=4)

    plt.imshow(rgb_instance)
    plt.show()

    plt.imshow(rgb_with_bbox)
    plt.show()


if __name__ == '__main__':
    train, test = get_all_available_by_category("bottom_cabinet")
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
