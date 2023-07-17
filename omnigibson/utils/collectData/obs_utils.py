import numpy as np


def get_bounding_box(seg_instance, obj_id):
    '''
    Returns x_min, y_min, x_extend, y_extend of the object with object_id
    Need to convert float and return in int
    '''
    if obj_id not in seg_instance:
        return None
    mask = seg_instance == obj_id
    # get the minimum and maximum coordinates of the object
    x_min = np.min(np.where(mask)[1])
    x_max = np.max(np.where(mask)[1])
    y_min = np.min(np.where(mask)[0])
    y_max = np.max(np.where(mask)[0])
    # get the extend of the object
    x_extend = x_max - x_min
    y_extend = y_max - y_min

    return [int(x_min), int(y_min), int(x_extend), int(y_extend)]
    

def save_link_level_obs(obj, obj_id, opened_links, prefix, fpath, cam):
    '''
    # needs to rewrite, original version lost @0704
    Returns the link-level observation of openable object
    Returns a dict, primary keys are links of obj (including "object")
    each corresponds to a dict with "bbox", "is_open", "is_prismatic"
    Returns None if ??? (need to recall condition) 
        - obj is not visible in the camera
    '''
    pass
