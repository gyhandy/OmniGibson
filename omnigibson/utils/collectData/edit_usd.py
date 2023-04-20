import omnigibson as og
from omnigibson.scenes import Scene
from omnigibson.utils.asset_utils import decrypt_file, encrypt_file
from pxr import Vt, Gf
from omni.isaac.core.utils.stage import open_stage, get_current_stage, close_stage
from IPython import embed
import numpy as np


def flip_visual_AA_trans(model, link):
    # First decrypt file
    IN_PATH = f"/scr/OmniGibson/omnigibson/data/og_dataset_temp/og_dataset/objects/bottom_cabinet/{model}/usd/{model}.encrypted.usd"
    OUT_PATH = f"./temp.usd"
    decrypt_file(IN_PATH, OUT_PATH)

    # Open USD
    open_stage(OUT_PATH)
    stage = get_current_stage()
    prim = stage.GetDefaultPrim()

    metadata = prim.GetCustomData()
    # embed()

    # Update a Vt array
    arr = np.array(metadata['metadata']['link_bounding_boxes'][link]['visual']['axis_aligned']['transform'])
    arr[:3, 3] = -arr[:3, 3]
    for i in range(3):
        vec_vt = Gf.Vec4f(float(arr[i][0]), float(arr[i][1]),float(arr[i][2]), float(arr[i][3]))
        metadata['metadata']['link_bounding_boxes'][link]['visual']['axis_aligned']['transform'][i] = vec_vt
    # metadata[...][0] = vec_vt

    # Write metadata
    prim.SetCustomData(metadata)

    # Save the USD
    stage.Save()

    # Re-encrypt file
    encrypt_file(OUT_PATH, IN_PATH)

scene = Scene()
og.sim.import_scene(scene)
