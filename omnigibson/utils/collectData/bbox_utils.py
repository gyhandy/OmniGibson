# OmniGibson imports
import omnigibson as og
from omnigibson.scenes import Scene
from omnigibson.objects import DatasetObject
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
import matplotlib.patches as patches
from omni.isaac.synthetic_utils.visualization import colorize_bboxes
import numpy as np
from itertools import product
import copy
import trimesh

from omnigibson.utils.collectData.env_utils import (
    create_env_with_light, 
    fold_cloth, 
    basic_objects,
    sample_cam_pose,
)
from omnigibson.utils.collectData.openable_obj import get_all_available_by_category
from omnigibson.utils.collectData.edit_usd import flip_visual_AA_trans
# from omnigibson.utils.collectData.openPipeline import sample_cam_pose

gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_GLOBAL_CONTACT_REPORTING = True
# gm.USE_GPU_DYNAMICS = True

def pts_to_bbox(pts, cam, intrinsics):
    '''
    Input: 
        pts -- array of target points to convert
        cam -- viewer camera obj(used to get pos and orientation)
        intrinsics -- intrinsic matrix of cam
    Output:
        2d bbox of pts in form (x_min, y_min, x_max, y_max)
    '''

    cam_pos, cam_orn = cam.get_position_orientation()

    # Transfer to camera coordinate
    # TODO: try to vectorize this
    pts_in_cam = []
    for pt in pts:
        pts_in_cam.append(T.relative_pose_transform(pt, [0, 0, 0, 1], cam_pos, cam_orn)[0])
    pts_in_cam = np.array(pts_in_cam)

    # flip y, z coordinate in camera frame
    pts_in_cam[:, 1] *= -1
    pts_in_cam[:, 2] *= -1

    # project pts_in_cam to pixel coordinates
    pixels = []
    for pt_in_cam in pts_in_cam:
        pixel = intrinsics @ pt_in_cam
        pixel /= pixel[2]
        pixels.append(pixel)
    pixels = np.array(pixels)[:, :2]

    # compute bounding box
    x_min, y_min = np.min(pixels, axis=0)
    x_max, y_max = np.max(pixels, axis=0)

    return np.array([x_min, y_min, x_max, y_max])


def compute_intrinsics(cam, img_width):
    '''
    Now assumes squared image. 
    Need to pass in image height if non-square
    '''
    apert = cam.prim.GetAttribute("horizontalAperture").Get()
    focal_len_in_pixel = cam.focal_length * img_width / apert

    intrinsics = np.eye(3)
    intrinsics[0,0] = focal_len_in_pixel
    intrinsics[1,1] = focal_len_in_pixel
    intrinsics[0,2] = img_width / 2
    intrinsics[1,2] = img_width / 2

    return intrinsics


def center_extent_to_xy_minmax_3d(center, extent, orientation=np.array([0., 0., 0., 1.])):
    '''
    '''
    pts = []
    center = np.array(center)        
    # Normalize the quaternion
    q = trimesh.transformations.unit_vector(orientation)
    
    # Transform the quarternion from (x, y, z, w) to (w, x, y, z)
    q = np.roll(q, shift=1)

    # Create a transformation matrix from the quaternion
    T = trimesh.transformations.quaternion_matrix(q)

    for p in product([-0.5, 0.5], repeat=3):
        v = np.array(extent) * np.array(p)

        # Transform the vector using the matrix
        v_transformed = np.dot(T, np.hstack((v, 1)))[0:3]

        pts.append(center + v_transformed)

    return pts


def plot_pixels(pixels, obs, all=True):
    '''
    Helper func to plot bboxes from pixels
    If all=False, assume have 8 pixels, will plot 4 by 4
    '''
    if all:
        x_min, y_min = np.min(pixels, axis=0)
        x_max, y_max = np.max(pixels, axis=0)
    else:
        x_min, y_min = np.min(pixels[:4], axis=0)
        x_max, y_max = np.max(pixels[:4], axis=0)

    bbox_obs = obs['bbox_2d_loose']
    bbox = list(bbox_obs[0])
    bbox[-4:] = [x_min, y_min, x_max, y_max]
    bbox_obs[0] = tuple(bbox)
    img = colorize_bboxes(bboxes_2d_data=bbox_obs, bboxes_2d_rgb=obs["rgb"], num_channels=4)
    plt.imshow(img)
    plt.show()

    if all:
        return bbox_obs
    
    x_min, y_min = np.min(pixels[4:], axis=0)
    x_max, y_max = np.max(pixels[4:], axis=0)
    bbox[-4:] = [x_min, y_min, x_max, y_max]
    bbox_obs[0] = tuple(bbox)
    img = colorize_bboxes(bboxes_2d_data=bbox_obs, bboxes_2d_rgb=obs["rgb"], num_channels=4)
    plt.imshow(img)
    plt.show()

    return bbox_obs


def add_bboxes_to_obs(bbox_obs, ref, bboxes):
    '''
    bbox_obs: the current bbox obs to modify, assume uniqueID in increaing order
    ref: an item in bbox obs that the new bboxes shall share metadata with
    bboxes: list of x_min, y_min, x_max, y_max
    '''

    cur_max_id = list(bbox_obs[-1])[0]

    for bbox in bboxes:
        new_bbox = list(copy.deepcopy(ref))
        new_bbox[0] = cur_max_id + 1
        cur_max_id += 1
        new_bbox[1:5] = bbox
        bbox_obs = np.append(bbox_obs, np.array(tuple(new_bbox), dtype=ref.dtype))

    return bbox_obs

def get_all_link_3d_bboxes(obj):
    '''
    Returns a list of all link-level 3d-bboxes in obj as a dict
    '''
    link_names = list(obj.links.keys())
    all_3d_bboxes = {}

    for link_name in link_names:
        center, orientation, extent, _ = obj.get_base_aligned_bbox(link_name=link_name, visual=True, xy_aligned=True, fallback_to_aabb=False, link_bbox_type="axis_aligned")
        pts = center_extent_to_xy_minmax_3d(center, extent, orientation)
        all_3d_bboxes[link_name] = pts

    return all_3d_bboxes

def plot_all_bboxes(obj, cam, link=None):
    all_3d_bboxes = get_all_link_3d_bboxes(obj)
    intrinsics = compute_intrinsics(cam, 1024)
    for key in all_3d_bboxes.keys(): 
        all_3d_bboxes[key] = pts_to_bbox(all_3d_bboxes[key], cam, intrinsics)
    
    bboxes = []
    if not link:
        for key in all_3d_bboxes.keys(): 
            bboxes.append(all_3d_bboxes[key])
    else:
        bboxes.append(all_3d_bboxes[link])

    bbox_obs = cam.get_obs()["bbox_2d_loose"]

    if len(bbox_obs) == 0:
        og.log.info("No bbox obs, skipped")
        return None
    ref = bbox_obs[0]
    bbox_obs = add_bboxes_to_obs(bbox_obs, ref, bboxes)

    plot_bbox_on_rgb(bbox_obs, cam.get_obs()["rgb"], fpath=None, is_minmax=True)

    # img = colorize_bboxes(bboxes_2d_data=bbox_obs, bboxes_2d_rgb=cam.get_obs()["rgb"], num_channels=4)
    # plt.imshow(img)
    # plt.show()
    # embed()

def get_all_bboxes(obj, cam, bbox_obs=None, img_fpath=None):
    '''
    obj: the DatasetObject to get all bbox for its links
    cam: viewer camera
    bbox_obs: original bbox obs FOR THIS OBJECT -- will append link bbox to it
    img_fpath: path to save the rgb image. Default is None.
    '''
    # obtain 3d bboxes and camera information
    all_3d_bboxes = get_all_link_3d_bboxes(obj)
    intrinsics = compute_intrinsics(cam, 1024)

    # obtain 2d bboxes, in both dict and array form
    link_bboxes = {}
    link_array_bboxes = None
    for key in all_3d_bboxes.keys(): 
        link_bboxes[key] = pts_to_bbox(all_3d_bboxes[key], cam, intrinsics).clip(0, 1023)
        if link_array_bboxes is None:
            link_array_bboxes = link_bboxes[key]
        else:
            link_array_bboxes = np.vstack((link_array_bboxes, link_bboxes[key]))

    # union bboxes for object bbox
    obj_bbox = np.append(np.min(link_array_bboxes, axis=0)[:2], np.max(link_array_bboxes, axis=0)[2:])
    link_bboxes["object"] = obj_bbox

    if bbox_obs is None:
        bbox_obs = cam.get_obs()["bbox_2d_loose"]

    if len(bbox_obs) == 0:
        og.log.info("No bbox obs, skipped")
        return None
    
    ref = bbox_obs[0]
    bbox_obs = add_bboxes_to_obs(bbox_obs, ref, list(link_bboxes.values()))

    # if img_fpath is None:
    #     og.log.info("Not saving bbox image")
    # else:
    # # bbox_img = colorize_bboxes(bboxes_2d_data=bbox_obs[1:], bboxes_2d_rgb=cam.get_obs()["rgb"], num_channels=4)
    #     plot_bbox_on_rgb(bboxes=bbox_obs[1:], rgb=cam.get_obs()["rgb"], fpath=img_fpath, is_minmax=True)

    # TODO: rconvert xyminmax to cornerextent
    for key in link_bboxes:
        orig = link_bboxes[key]
        link_bboxes[key] = ((int(orig[0]), int(orig[1])),(int(orig[2]-orig[0]), int(orig[3]-orig[1])))

    # return link_bboxes, bbox_img
    return link_bboxes, bbox_obs[1:]

def new_get_all_bboxes(obj, cam):
    # obtain 3d bboxes and camera information
    all_3d_bboxes = get_all_link_3d_bboxes(obj)
    intrinsics = compute_intrinsics(cam, 1024)

    # obtain 2d bboxes, in both dict and array form
    link_bboxes = {}
    link_array_bboxes = None
    for key in all_3d_bboxes.keys(): 
        obj_name = obj.name.split("_")[0]
        link_bboxes[f"{obj_name}-{key}"] = pts_to_bbox(all_3d_bboxes[key], cam, intrinsics).clip(0, 1023)
        if link_array_bboxes is None:
            link_array_bboxes = link_bboxes[f"{obj_name}-{key}"]
        else:
            link_array_bboxes = np.vstack((link_array_bboxes, link_bboxes[f"{obj_name}-{key}"]))

    # union bboxes for object bbox
    obj_bbox = np.append(np.min(link_array_bboxes, axis=0)[:2], np.max(link_array_bboxes, axis=0)[2:])
    link_bboxes[obj_name] = obj_bbox

    bbox_obs = cam.get_obs()["bbox_2d_loose"]

    if len(bbox_obs) == 0:
        og.log.info("No bbox obs, skipped")
        return None
    ref = bbox_obs[0]
    bbox_obs = add_bboxes_to_obs(bbox_obs, ref, list(link_bboxes.values()))

    bbox_img = colorize_bboxes(bboxes_2d_data=bbox_obs[1:], bboxes_2d_rgb=cam.get_obs()["rgb"], num_channels=4)

    # TODO: convert xyminmax to cornerextent
    for key in link_bboxes:
        orig = link_bboxes[key]
        link_bboxes[key] = ((int(orig[0]), int(orig[1])),(int(orig[2]-orig[0]), int(orig[3]-orig[1])))
    # print(link_bboxes)
    # plt.imshow(bbox_img)
    # plt.show()
    return link_bboxes, bbox_img


def xy_minmax_to_center_extend_2d(orig):
    return ((int(orig[0]), int(orig[1])),(int(orig[2]-orig[0]), int(orig[3]-orig[1])))


def plot_bbox_on_rgb(bboxes, rgb, fpath, is_minmax=True):
    # This method is written for the new form of bbox observation
    # ('semanticId', '<u4'), ('x_min', '<i4'), ('y_min', '<i4'), ('x_max', '<i4'), ('y_max', '<i4'), ('occlusionRatio', '<f4')
    # if is_minmax is False, indicating original position for x_max, y_max now have x_extent, y_extent

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(rgb)

    # Draw a rectangle
    i = 0
    colors = ["red", "blue", "yellow", "green", "black", "purple", "gray"]
    for bbox in bboxes:
        (x_min, y_min) = (bbox[1], bbox[2])
        if is_minmax:
            x_extent, y_extent = bbox[3]-bbox[1], bbox[4]-bbox[2]
        else:
            x_extent, y_extent = bbox[3], bbox[4]
        rect = patches.Rectangle((x_min, y_min), x_extent, y_extent, linewidth=1, edgecolor=colors[i], facecolor='none')
        i+=1
        i = i % len(colors)

        # Add the rectangle to the axes
        ax.add_patch(rect)

    # Save the figure to a PNG file
    plt.savefig(fpath)
    # plt.imsave(fpath, rgb)
    # plt.show()

    plt.close(fig)

if __name__ == '__main__':
    ########## Initiate env and camera #########
    # #model="lwjdmj",##############

    env, cam = create_env_with_light()
    cab0, cab1, cupcake, laptop, table, shirt, carpet = basic_objects()
    train, test = get_all_available_by_category('bottom_cabinet', use_avg_spec=False)

    og.sim.stop()
    og.sim.import_object(cab0)
    og.sim.play()
    for _ in range(30): og.sim.step()

    embed()
    env.close()
    

    # for obj in train:
    #     og.sim.stop()
    #     og.sim.import_object(obj)
    #     og.sim.play()
    #     for _ in range(30): og.sim.step()

    #     embed()

    #     og.sim.stop()
    #     scene = Scene()
    #     og.sim.import_scene(scene)
    #     og.sim.viewer_camera.add_modality('bbox_2d_loose')
    #     cam = VisionSensor(
    #         prim_path="/World/viewer_camera",
    #         name="camera",
    #         modalities=["rgb", "seg_instance","bbox_2d_loose", "bbox_2d_tight"], #"depth_linear", "seg_instance", "bbox_2d_tight", "bbox_3d", "camera"],
    #         image_height=1024,
    #         image_width=1024,
    #     )
    #     cam.initialize()
    #     og.sim.enable_viewer_camera_teleoperation()

        # vespxk
        # ujniob, ybntlp

    # for obj in train[10:]:
    #     og.sim.stop()
    #     og.sim.import_object(obj)
    #     og.log.info(f"visualizing model {obj.name}")
    #     og.sim.play()
    #     for _ in range(30): og.sim.step()
        
    #     for _ in range(2):
    #         cam_pos = sample_cam_pose() + obj.get_position()
    #         set_camera_view(cam_pos, obj.get_position(), camera_prim_path="/World/viewer_camera", viewport_api=None)
    #         for _ in range(30): og.sim.render()
    #         # plot_all_bboxes(obj, cam)
    #         get_all_bboxes(obj, cam)

    #     for _ in range(3):
    #         suc, links = obj.states[Open].set_value(True, fully=True)
    #         # embed()
    #         for _ in range(30): og.sim.step()
    #         cam_pos = sample_cam_pose() + obj.get_position()
    #         set_camera_view(cam_pos, obj.get_position(), camera_prim_path="/World/viewer_camera", viewport_api=None)
    #         for _ in range(30): og.sim.render()
    #         # plot_all_bboxes(obj, cam)
    #         get_all_bboxes(obj, cam)

    #     og.sim.scene.remove_object(obj)
    #     for _ in range(30): og.sim.step()

    
    embed()
    env.close()