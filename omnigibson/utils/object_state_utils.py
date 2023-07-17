import cv2
import numpy as np

from IPython import embed
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull, distance_matrix

import omnigibson as og
from omnigibson.macros import create_module_macros, Dict, gm
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contact_bodies import ContactBodies
from omnigibson.utils import sampling_utils
from omnigibson.utils.constants import PrimType
from omnigibson.utils.ui_utils import debug_breakpoint
import omnigibson.utils.transform_utils as T


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.ON_TOP_RAY_CASTING_SAMPLING_PARAMS = Dict({
    "bimodal_stdev_fraction": 1e-6,
    "bimodal_mean_fraction": 1.0,
    "aabb_offset": 0.01,
    "max_sampling_attempts": 50,
})

m.INSIDE_RAY_CASTING_SAMPLING_PARAMS = Dict({
    "bimodal_stdev_fraction": 0.4,
    "bimodal_mean_fraction": 0.5,
    "aabb_offset": 0.0,
    "max_sampling_attempts": 100,
})

m.UNDER_RAY_CASTING_SAMPLING_PARAMS = Dict({
    "bimodal_stdev_fraction": 1e-6,
    "bimodal_mean_fraction": 0.5,
    "aabb_offset": 0.01,
    "max_sampling_attempts": 50,
})


def sample_kinematics(
    predicate,
    objA,
    objB,
    use_ray_casting_method=True,
    max_trials=10,
    z_offset=0.05,
    skip_falling=False,
    sample_link_name=None,
):
    """
    Samples the given @predicate kinematic state for @objA with respect to @objB

    Args:
        predicate (str): Name of the predicate to sample, e.g.: "onTop"
        objA (StatefulObject): Object whose state should be sampled. e.g.: for sampling a microwave
            on a cabinet, @objA is the microwave
        objB (StatefulObject): Object who is the reference point for @objA's state. e.g.: for sampling
            a microwave on a cabinet, @objB is the cabinet
        use_ray_casting_method (bool): Whether to use raycasting for sampling or not
        max_trials (int): Number of attempts for sampling
        z_offset (float): Z-offset to apply to the sampled pose
        skip_falling (bool): Whether to let @objA fall after its position is sampled or not
        (Yihe added)sample_link_name (str): if only sampling for specific link in objB (used for Inside)

    Returns:
        bool: True if successfully sampled, else False
    """
    assert z_offset > 0.5 * 9.81 * (og.sim.get_physics_dt() ** 2) + 0.02,\
        f"z_offset {z_offset} is too small for the current physics_dt {og.sim.get_physics_dt()}"

    # Run import here to avoid circular imports
    # No supporting surface annotation found, fallback to use ray-casting
    from omnigibson.objects.dataset_object import DatasetObject
    if (
        not isinstance(objB, DatasetObject) or
        len(objB.supporting_surfaces) == 0 or
        predicate not in objB.supporting_surfaces
    ):
        use_ray_casting_method = True

    # Wake objects accordingly and make sure both are kept still
    objA.wake()
    objB.wake()

    objA.keep_still()
    objB.keep_still()

    # Save the state of the simulator
    state = og.sim.dump_state()

    # Attempt sampling
    for i in range(max_trials):
        pos = None
        if hasattr(objA, "orientations") and objA.orientations is not None:
            orientation = objA.sample_orientation()
        else:
            orientation = np.array([0, 0, 0, 1.0])

        # always place new object in up orientation
        orientation = np.array([0, 0, 0, 1.0])

        # Orientation needs to be set for stable_z_on_aabb to work correctly
        # Position needs to be set to be very far away because the object's
        # original position might be blocking rays (use_ray_casting_method=True)
        old_pos = np.array([100, 100, 10])
        objA.set_position_orientation(old_pos, orientation)
        objA.keep_still()
        # We also need to step physics to make sure the pose propagates downstream (e.g.: to Bounding Box computations)
        og.sim.step_physics()

        # This would slightly change because of the step_physics call.
        old_pos, orientation = objA.get_position_orientation()

        if use_ray_casting_method:
            if predicate == "onTop":
                params = m.ON_TOP_RAY_CASTING_SAMPLING_PARAMS
            elif predicate == "inside":
                params = m.INSIDE_RAY_CASTING_SAMPLING_PARAMS
            elif predicate == "under":
                params = m.UNDER_RAY_CASTING_SAMPLING_PARAMS
            else:
                raise ValueError(f"predicate must be onTop, under or inside in order to use ray casting-based "
                                 f"kinematic sampling, but instead got: {predicate}")

            # Run import here to avoid circular imports
            from omnigibson.objects.dataset_object import DatasetObject
            if isinstance(objA, DatasetObject) and objA.prim_type == PrimType.RIGID:
                # Retrieve base CoM frame-aligned bounding box parallel to the XY plane
                parallel_bbox_center, parallel_bbox_orn, parallel_bbox_extents, _ = objA.get_base_aligned_bbox(
                    xy_aligned=True
                )
            else:
                aabb_lower, aabb_upper = objA.states[AABB].get_value()
                parallel_bbox_center = (aabb_lower + aabb_upper) / 2.0
                parallel_bbox_orn = np.array([0.0, 0.0, 0.0, 1.0])
                parallel_bbox_extents = aabb_upper - aabb_lower

            if predicate == "under":
                start_points, end_points = sampling_utils.sample_raytest_start_end_symmetric_bimodal_distribution(
                    obj=objB,
                    num_samples=1,
                    axis_probabilities=[0, 0, 1],
                    **params,
                )
                sampling_results = sampling_utils.sample_cuboid_on_object(
                    obj=None,
                    start_points=start_points,
                    end_points=end_points,
                    ignore_objs=[objB],
                    cuboid_dimensions=parallel_bbox_extents,
                    refuse_downwards=True,
                    undo_cuboid_bottom_padding=True,
                    max_angle_with_z_axis=0.17,
                    hit_proportion=0.0,  # rays will NOT hit the object itself, but the surface below it.
                )
            elif predicate == "onTop":
                sampling_results = sampling_utils.sample_cuboid_on_object_symmetric_bimodal_distribution(
                    objB,
                    num_samples=1,
                    axis_probabilities=[0, 0, 1],
                    cuboid_dimensions=parallel_bbox_extents,
                    refuse_downwards=True,
                    undo_cuboid_bottom_padding=True,
                    max_angle_with_z_axis=0.17,
                    sample_link_name=objB.root_link_name, # only sample on top of root link
                    **params,
                )
            elif predicate == "inside" and sample_link_name is not None:
                # sample inside a spefic link
                sampling_results = sampling_utils.sample_cuboid_on_object_symmetric_bimodal_distribution(
                    objB,
                    num_samples=1,
                    axis_probabilities=[0, 0, 1],
                    cuboid_dimensions=parallel_bbox_extents,
                    refuse_downwards=True,
                    undo_cuboid_bottom_padding=True,
                    max_angle_with_z_axis=0.17,
                    sample_link_name=sample_link_name,
                    **params,
                )
            else:
                sampling_results = sampling_utils.sample_cuboid_on_object_symmetric_bimodal_distribution(
                    objB,
                    num_samples=1,
                    axis_probabilities=[0, 0, 1],
                    cuboid_dimensions=parallel_bbox_extents,
                    refuse_downwards=True,
                    undo_cuboid_bottom_padding=True,
                    max_angle_with_z_axis=0.17,
                    **params,
                )

            sampled_vector = sampling_results[0][0]
            sampled_quaternion = sampling_results[0][2]

            sampling_success = sampled_vector is not None
            if sampling_success:
                # Move the object from the original parallel bbox to the sampled bbox
                parallel_bbox_rotation = R.from_quat(parallel_bbox_orn)
                sample_rotation = R.from_quat(sampled_quaternion)
                original_rotation = R.from_quat(orientation)

                # The additional orientation to be applied should be the delta orientation
                # between the parallel bbox orientation and the sample orientation
                additional_rotation = sample_rotation * parallel_bbox_rotation.inv()
                combined_rotation = additional_rotation * original_rotation
                orientation = combined_rotation.as_quat()

                # The delta vector between the base CoM frame and the parallel bbox center needs to be rotated
                # by the same additional orientation
                diff = old_pos - parallel_bbox_center
                rotated_diff = additional_rotation.apply(diff)
                pos = sampled_vector + rotated_diff
        else:
            random_idx = np.random.randint(len(objB.supporting_surfaces[predicate].keys()))
            objB_link_name = list(objB.supporting_surfaces[predicate].keys())[random_idx]
            random_height_idx = np.random.randint(len(objB.supporting_surfaces[predicate][objB_link_name]))
            height, height_map = objB.supporting_surfaces[predicate][objB_link_name][random_height_idx]
            obj_half_size = np.max(objA.aabb_extent) / 2 * 100
            obj_half_size_scaled = np.array([obj_half_size / objB.scale[1], obj_half_size / objB.scale[0]])
            obj_half_size_scaled = np.ceil(obj_half_size_scaled).astype(np.int)
            height_map_eroded = cv2.erode(height_map, np.ones(obj_half_size_scaled, np.uint8))

            valid_pos = np.array(height_map_eroded.nonzero())
            if valid_pos.shape[1] != 0:
                random_pos_idx = np.random.randint(valid_pos.shape[1])
                random_pos = valid_pos[:, random_pos_idx]
                y_map, x_map = random_pos
                y = y_map / 100.0 - 2
                x = x_map / 100.0 - 2
                z = height

                pos = np.array([x, y, z])
                pos *= objB.scale

                # the supporting surface is defined w.r.t to the link frame, so we need to convert it into
                # the world frame
                link_pos, link_quat = objB.links[objB_link_name].get_position_orientation()
                pos = T.quat2mat(link_quat).dot(pos) + np.array(link_pos)
                # Get the combined AABB.
                lower, _ = objA.states[AABB].get_value()
                # Move the position to a stable Z for the object.
                pos[2] += objA.get_position()[2] - lower[2]

        if pos is None:
            success = False
        else:
            pos[2] += z_offset
            objA.set_position_orientation(pos, orientation)
            objA.keep_still()

            og.sim.step_physics()
            objA.keep_still()
            success = len(objA.states[ContactBodies].get_value()) == 0

        if gm.DEBUG:
            debug_breakpoint(f"sample_kinematics: {success}")

        if success:
            break
        else:
            og.sim.load_state(state)

    if success and not skip_falling:
        objA.set_position_orientation(pos, orientation)
        objA.keep_still()

        # Let it fall for 0.2 second
        for _ in range(int(0.2 / og.sim.get_physics_dt())):
            og.sim.step_physics()
            if len(objA.states[ContactBodies].get_value()) > 0:
                break

        objA.keep_still()

        # Render at the end
        og.sim.render()

    return success


def sample_cloth_on_rigid(obj, other, max_trials=10, z_offset=0.05, randomize_xy=True):
    """
    Samples the cloth object @obj on the rigid object @other

    Args:
        obj (StatefulObject): Object whose state should be sampled. e.g.: for sampling a bed sheet on a rack,
            @obj is the bed sheet
        other (StatefulObject): Object who is the reference point for @obj's state. e.g.: for sampling a bed sheet
            on a rack, @other is the rack
        max_trials (int): Number of attempts for sampling
        z_offset (float): Z-offset to apply to the sampled pose
        randomize_xy (bool): Whether to randomize the XY position of the sampled pose. If False, the center of @other
            will always be used.

    Returns:
        bool: True if successfully sampled, else False
    """
    assert z_offset > 0.5 * 9.81 * (og.sim.get_physics_dt() ** 2) + 0.02,\
        f"z_offset {z_offset} is too small for the current physics_dt {og.sim.get_physics_dt()}"

    if not (obj.prim_type == PrimType.CLOTH and other.prim_type == PrimType.RIGID):
        raise ValueError("sample_cloth_on_rigid requires obj1 is cloth and obj2 is rigid.")

    state = og.sim.dump_state(serialized=False)

    # Reset the cloth
    obj.root_link.reset()

    obj_aabb_low, obj_aabb_high = obj.states[AABB].get_value()
    other_aabb_low, other_aabb_high = other.states[AABB].get_value()

    # z value is always the same: the top-z of the other object + half the height of the object to be placed + offset
    z_value = other_aabb_high[2] + (obj_aabb_high[2] - obj_aabb_low[2]) / 2.0 + z_offset

    if randomize_xy:
        # Sample a random position in the x-y plane within the other object's AABB
        low = np.array([other_aabb_low[0], other_aabb_low[1], z_value])
        high = np.array([other_aabb_high[0], other_aabb_high[1], z_value])
    else:
        # Always sample the center of the other object's AABB
        low = np.array([(other_aabb_low[0] + other_aabb_high[0]) / 2.0,
                        (other_aabb_low[1] + other_aabb_high[1]) / 2.0,
                        z_value])
        high = low

    for _ in range(max_trials):
        # Sample a random position
        pos = np.random.uniform(low, high)
        # Sample a random orientation in the z-axis
        orn = T.euler2quat(np.array([0., 0., np.random.uniform(0, np.pi * 2)]))

        obj.set_position_orientation(pos, orn)
        obj.root_link.reset()
        obj.keep_still()

        og.sim.step_physics()
        success = len(obj.states[ContactBodies].get_value()) == 0

        if success:
            break
        else:
            og.sim.load_state(state)

    if success:
        # Let it fall for 0.2 second always to let the cloth settle
        for _ in range(int(0.2 / og.sim.get_physics_dt())):
            og.sim.step_physics()

        obj.keep_still()

        # Render at the end
        og.sim.render()

    return success
