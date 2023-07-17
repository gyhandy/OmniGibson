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
    get_objects_by_categories,
)
from omnigibson.utils.collectData.bbox_utils import get_all_bboxes
from omnigibson.utils.collectData.env_utils import (
    create_env_with_light,
    sample_cam_pose,
    basic_objects
)

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True
import json


class SceneGraphNode:
    def __init__(self, obj) -> None:
        self.object = obj
    
class SceneGraphEdge:
    def __init__(self, start_node, end_node, edge_type) -> None:
        self.start_node = start_node
        self.end_node = end_node
        self.edge_type = edge_type

class SceneGraph:
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

    def add_edge(self, edge:SceneGraphEdge):
        self.edges.append(edge)

    def add_node(self, node:SceneGraphNode):
        self.nodes.append(node)

    def get_subgraph(self, node_subset):
        subgraph = SceneGraph()
        
        for node in node_subset:
            subgraph.add_node(node)
        for edge in self.edges:
            if edge.start_node in node_subset and edge.end_node in node_subset:
                subgraph.add_edge(edge)
    
        return subgraph


def create_graph_from_scene():
    pass

def create_nodes_from_scene(graph:SceneGraph):
    for obj in og.sim.scene.objects:
        if isinstance(obj, DatasetObject):
            node = SceneGraphNode(obj)
            graph.add_node(node)

def create_nodes_from_objects(graph, objs):
    for obj in objs:
        node = SceneGraphNode(obj)
        graph.add_node(node)


def create_edges_from_scene(nodes):

    # for every pair of nodes
    
        # for each possible relationship

            # check if that relationship holds


    pass

def scene_graph_to_json(graph: SceneGraph):
    graph_dict = {}

    graph_dict["objects"] = {}
    for node in graph.nodes:
        graph_dict["objects"][node.object.category] = {
            "category": node.object.category,
            "position": tuple(node.object.get_position()),
            "orientation": tuple(node.object.get_orientation()),
            "physical_attribute": "hard"
        }

    graph_dict["relations"] = []
    for edge in graph.edges:
        graph_dict["relations"].append(
            (edge.start_node.object.category, edge.end_node.object.category, edge.edge_type)
        )

    with open("./collected_data/scene_info_2.json", "w") as f:
        json.dump(graph_dict, f, indent=4)
    f.close()

    rgb = og.sim.viewer_camera.get_obs()["rgb"][:,:,:3]
    plt.imsave("./collected_data/scene_image_2.png", rgb)


if __name__ == "__main__":
    
    # Load a predefined scene
    scenes = get_available_og_scenes()
    scene_type = "InteractiveTraversableScene"
    scene_model = list(scenes)[3]
    # scene_model = list(scenes)[2]
    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
            # "load_object_categories": ["floors", "walls", "ceilings", "breakfast_table", "swivel_chair", "trash_can"]
        },
    }


    env = og.Environment(configs=cfg)
    cam_mover = og.sim.enable_viewer_camera_teleoperation()

    embed()

