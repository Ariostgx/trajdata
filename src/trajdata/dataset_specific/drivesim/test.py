from typing import Dict

import numpy as np
import nvmap_utils as nvutils

# This is from the <geoReference> tag from the .xodr file for Endeavor.
LATLONALT_ORIGIN_ENDEAVOR = np.array([[37.37852062996696, -121.9596180846297, 0.0]])


def convert_to_DS(poses: np.ndarray):
    """_summary_

    Args:
        poses (np.ndarray): x, y, h states of each agent at each time, relative to the ego vehicle's state at time t=0. (N, T, 3)
    """
    # wgs84 = coutils.LocalToWGS84((0, 0, 0), (37.37852062996696, -121.9596180846297, 0.0))
    # wgs84_2 = coutils.ECEFtoWGS84(nvutils.GLOBAL_BASE_POSE_ENDEAVOR[:3, -1])

    world_from_map_ft = nvutils.lat_lng_alt_2_ecef(
        LATLONALT_ORIGIN_ENDEAVOR, np.array([[1, 0, 0]]), np.array([[0]])
    )

    fps = 30
    N, T = poses.shape[:2]
    x = poses[..., 0]
    y = poses[..., 1]
    heading = poses[..., 2]

    c = np.cos(heading)
    s = np.sin(heading)
    T_mat = np.tile(np.eye(4), (N, T, 1, 1))
    T_mat[..., 0, 0] = c
    T_mat[..., 0, 1] = -s
    T_mat[..., 1, 0] = s
    T_mat[..., 1, 1] = c
    T_mat[..., 0, 3] = x
    T_mat[..., 1, 3] = y
    # TODO: Some height for ray-casting down to road?
    # T_mat[..., 2, 3] = 0

    ecef_traj_poses = np.matmul(world_from_map_ft[np.newaxis, np.newaxis], T_mat)
    gps_traj_poses = nvutils.ecef_2_lat_lng_alt(
        ecef_traj_poses.reshape((N * T, 4, 4)), earth_model="WGS84"
    )
    lat_lng_alt, orientation_axis, orientation_angle = gps_traj_poses

    out_dict = {
        "timestamps": np.linspace(0, T / fps, 315),
        "track_ids": np.array(["ego", "769"]),
        # 'bbox_lwh' : np.array([[4.387, 1.907, 1.656], [4.387, 1.907, 1.656]]),
        # 'ecef_poses' : ecef_traj_poses,
        "pose_valid": np.ones(
            (ecef_traj_poses.shape[0], ecef_traj_poses.shape[1]), dtype=bool
        ),
        "gps_lat_lng_alt": lat_lng_alt.reshape((N, T, 3)),
        "gps_orientation_axis": orientation_axis.reshape((N, T, 3)),
        "gps_orientation_angle_degrees": orientation_angle.reshape((N, T, 1)),
    }

    return out_dict


def main():
    poses = np.zeros((2, 315, 3))
    poses[0, :, 0] = np.linspace(-505, -519, 315)
    poses[0, :, 1] = np.linspace(-1019, -866, 315)
    poses[0, :, 2] = np.linspace(np.pi / 2, 5 * np.pi / 8, 315)

    out_dict = convert_to_DS(poses)

    with np.load(
        "/home/bivanovic/projects/drivesim-ov/source/extensions/omni.drivesim.dl_traffic_model/data/example_trajectories.npz"
    ) as data:
        out_dict["gps_lat_lng_alt"][1] = data["gps_lat_lng_alt"][1]
        out_dict["gps_orientation_axis"][1] = data["gps_orientation_axis"][1]
        out_dict["gps_orientation_angle_degrees"][1] = data[
            "gps_orientation_angle_degrees"
        ][1]

    np.savez(
        "/home/bivanovic/projects/drivesim-ov/source/extensions/omni.drivesim.dl_traffic_model/data/test.npz",
        **out_dict
    )


if __name__ == "__main__":
    main()
