import copy
import glob
import json
import os
import shutil
import sys
from collections import OrderedDict, deque

import cv2
import matplotlib as mpl

# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon
from matplotlib.path import Path

# NOTE: NVMap documentation https://nvmapspec.drive.nvda.io/map_layer_format/index.html

# transformation matrix of the base pose of the global cooridinate system (defined in ECEF)
#       will be used to transform each road segment to global frame.
# rivermark is used for autolabels
GLOBAL_BASE_POSE_RIVERMARK = np.array(
    [
        [
            0.7661314567952979,
            -0.47780431092164843,
            -0.42982046411659275,
            -2684537.8381108316,
        ],
        [
            0.06759529959471867,
            0.7249870917843614,
            -0.6854375188292178,
            -4305029.705512633,
        ],
        [
            0.6391192896333319,
            0.49608140179888094,
            0.5877327423309361,
            3852303.4349504006,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# endeavor can be used for map if desired
GLOBAL_BASE_POSE_ENDEAVOR = np.array(
    [
        [
            0.892451945618358,
            0.11765375230214345,
            -0.43553084773783024,
            -2686874.9549495797,
        ],
        [
            -0.40233607907375113,
            0.644307257120398,
            -0.6503797643665964,
            -4305568.426410184,
        ],
        [
            0.2040960661981692,
            0.7556624596942837,
            0.6223496145826857,
            3850100.0530229895,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# this is the one that will be used for the map and trajectories
GLOBAL_BASE_POSE = GLOBAL_BASE_POSE_ENDEAVOR
# GLOBAL_BASE_POSE = GLOBAL_BASE_POSE_RIVERMARK

MAX_ENDEAVOR_FRAME = 3613

RIVERMARK_VIZ = False
# 0 : drivable only
# 1 : + ego
# 2 : + lines
# 3 : + other non-extra cars
# 4 : + all cars
RIVERMARK_VIZ_LAYER = 4
NVCOLOR = (118.0 / 255, 185.0 / 255, 0.0 / 255)
NVCOLOR2 = (55.0 / 255, 225.0 / 255, 0.0 / 255)
# EXTRA_DYN_LWH = [5.087, 2.307, 1.856]
EXTRA_DYN_LWH = [4.387, 1.907, 1.656]

NUSC_EXPAND_LEN = 0.0  # 2.0 # meters to expand drivable area outward

AUTOLABEL_DT = 0.1  # sec

NVMAP_LAYERS = {"lane_dividers_v1", "lanes_v1", "lane_channels_v1"}
SEGMENT_GRAPH_LAYER = "segment_graph_v1"

# maps track ids (which occur first in time) to other tracks that are
#       actually the same object
# NOTE: each association list must be sorted in temporal order
TRACK_ASSOC = {
    # frame 3000 - 3545 sequence
    2486: [2584, 2669],
    2546: [2618],
    2489: [2558],
    2496: [2553],
    2515: [2578],
    2525: [2615],
    2555: [2609],
    2546: [2618, 2673],
    # 2491 : [2566, 2631],
    2491: [2631],
    # 2566 : [2631],
    2484: [2692, 2729, 2745, 2766],
    2686: [2811],
    2618: [2673],
    2592: [2719],
    2251: [2847],
    2269: [2734, 2853],
    1280: [2768, 2790],
    2631: [2685],
    2561: [2699],
    # frame 570 - 990 sequences
    1298: [1360],
    1282: [1362],
    1241: [1366],
    1305: [1398],
    1286: [1377],
    1327: [1409],
    1444: [1576],
    1326: [1455],
    1376: [1425, 1435, 1444],
    1345: [1445, 1456, 1476],
    1346: [1477, 1491],
    # frame 270 - 600
    1067: [1193],
    1109: [1206],
    1126: [1229],
    1127: [1234],
    1137: [1255],
    1150: [1251],
    1168: [1246],
    1184: [1269],
    1203: [1225],
    1196: [1307],
    1211: [1315],
    1191: [1297],
    1183: [1276],
    1188: [1274],
    1178: [1250],
    # frame 1440 - 1860
    1717: [1761],
    1712: [1778],
    1704: [1786],
    1733: [1803],
    1708: [1722, 1811],
    1674: [1732, 1740],
    1760: [1837],
    1824: [1836],
    1793: [1872],
    1812: [1869],
    1748: [1812],
    1826: [1864],
    1723: [1756],
    # rivermark
    9409: [9517],
}
# false positive (or extrapolation is bad)
# 2566
TRACK_REMOVE = {
    2676,
    2687,
    2590,
    2833,
    1199,
    9413,
    9449,
    9437,
    9589,
    2588,
    2669,
    2562,
    2516,
    2566,
}  # 9449 is rivermark dynamic, and 9437 is rivermark trash can


def check_single_veh_coll(traj_tgt, lw_tgt, traj_others, lw_others):
    """
    Checks if the target trajectory collides with each of the given other trajectories.

    Assumes all trajectories and attributes are UNNORMALIZED. Handles nan frames in traj_others by simply skipping.

    :param traj_tgt: (T x 4)
    :param lw_tgt: (2, )
    :param traj_others: (N x T x 4)
    :param lw_others: (N x 2)

    :returns veh_coll: (N)
    :returns coll_time: (N)
    """
    from shapely.geometry import Polygon

    NA, FT, _ = traj_others.shape

    veh_coll = np.zeros((NA, FT), dtype=np.bool)
    poly_cache = dict()  # for the tgt polygons since used many times
    for aj in range(NA):
        for t in range(FT):
            # compute iou
            if t not in poly_cache:
                ai_state = traj_tgt[t, :]
                if np.sum(np.isnan(ai_state)) > 0:
                    continue
                ai_corners = get_corners(ai_state, lw_tgt)
                ai_poly = Polygon(ai_corners)
                poly_cache[t] = ai_poly
            else:
                ai_poly = poly_cache[t]

            aj_state = traj_others[aj, t, :]
            if np.sum(np.isnan(aj_state)) > 0:
                continue
            aj_corners = get_corners(aj_state, lw_others[aj])
            aj_poly = Polygon(aj_corners)
            cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
            if cur_iou > 0.02:
                veh_coll[aj, t] = True

    return veh_coll


def plt_color(i):
    clist = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return clist[i]


def get_rot(h):
    return np.array(
        [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
    )


def get_corners(box, lw):
    l, w = lw
    simple_box = np.array(
        [
            [-l / 2.0, -w / 2.0],
            [l / 2.0, -w / 2.0],
            [l / 2.0, w / 2.0],
            [-l / 2.0, w / 2.0],
        ]
    )
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box


def plot_box(box, lw, color="g", alpha=0.7, no_heading=False):
    l, w = lw
    h = np.arctan2(box[3], box[2])
    simple_box = get_corners(box, lw)

    arrow = np.array(
        [
            box[:2],
            box[:2] + l / 2.0 * np.array([np.cos(h), np.sin(h)]),
        ]
    )

    plt.fill(
        simple_box[:, 0],
        simple_box[:, 1],
        color=color,
        edgecolor="k",
        alpha=alpha,
        linewidth=1.0,
        zorder=3,
    )
    if not no_heading:
        # plt.plot(arrow[:, 0], arrow[:, 1], color, alpha=1.0)
        plt.plot(arrow[:, 0], arrow[:, 1], "k", alpha=alpha, zorder=3)


def create_video(img_path_form, out_path, fps):
    """
    Creates a video from a format for frame e.g. 'data_out/frame%04d.png'.
    Saves in out_path.
    """
    import subprocess

    # if RIVERMARK_VIZ:
    #     ffmpeg_cmd = ['ffmpeg', '-y', '-i', img_path_form,
    #                 '-vf', 'transpose=2', img_path_form]
    #     subprocess.run(ffmpeg_cmd)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-r",
        str(fps),
        "-i",
        img_path_form,
        "-vcodec",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    subprocess.run(ffmpeg_cmd)


def debug_viz_vid(
    seg_dict,
    prefix,
    poses,
    poses_valid,
    poses_lwh,
    comp_out_path="./out/dev_nvmap",
    fps=10,
    subsamp=3,
    pose_ids=None,
    **kwargs,
):
    poses = poses[:, ::subsamp]
    poses_valid = poses_valid[:, ::subsamp]
    T = poses.shape[1]
    out_dir = os.path.join(comp_out_path, prefix)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for t in range(T):
        print("rendering frame %d..." % (t))
        debug_viz_segs(
            seg_dict,
            "frame_%06d" % (t),
            poses[:, t : t + 1],
            poses_valid[:, t : t + 1],
            poses_lwh,
            comp_out_path=out_dir,
            pose_ids=pose_ids,
            ego_traj=poses[0],
            **kwargs,
        )
    create_video(os.path.join(out_dir, "frame_%06d.jpg"), out_dir + ".mp4", fps)


def debug_viz_segs(
    seg_dict,
    prefix,
    poses=None,
    poses_valid=None,
    poses_lwh=None,
    pose_ids=None,
    comp_out_path="./out/dev_nvmap",
    extent=80,
    grid=True,
    show_ticks=True,
    dpi=100,
    ego_traj=None,
):
    """
    Visualize segments in the given dictionary.

    :param seg_dict: segments dictionary
    :param prefix: prefix to save figure
    :param poses: NxTx4x4 trajectory for N vehicles that will be plotted if given.
    """
    if not os.path.exists(comp_out_path):
        os.makedirs(comp_out_path)

    # fig = plt.figure()
    fig = plt.figure(dpi=dpi)

    origins = []
    arr_len = 10.0
    for _, seg in seg_dict.items():
        if poses is not None:
            dist2ego = np.linalg.norm(seg.local2world[:2, -1] - poses[0, 0, :2, -1])
            if dist2ego > 200:
                continue
        # plot layers
        if "drivable_area" in seg.layers:
            # draw drivable area first so under everything else
            for drivable_poly in seg.layers["drivable_area"]:
                polypatch = Polygon(
                    drivable_poly[:, :2], color="darkgray", alpha=1.0, linestyle="-"
                )
                # linewidth=2)
                plt.gca().add_patch(polypatch)

        if RIVERMARK_VIZ and RIVERMARK_VIZ_LAYER < 2:
            # only drivable area
            continue

        for layer_k, layer_v in seg.layers.items():
            if layer_k in {"lane_dividers", "lane_divider"}:
                for lane_div in layer_v:
                    polyline = (
                        lane_div.polyline
                        if isinstance(lane_div, LaneDivider)
                        else lane_div
                    )
                    linepatch = PathPatch(
                        Path(polyline[:, :2]), fill=False, color="gold", linestyle="-"
                    )
                    # linewidth=2)
                    plt.gca().add_patch(linepatch)
            elif layer_k in {"road_dividers", "road_divider"}:
                for road_div in layer_v:
                    linepatch = PathPatch(
                        Path(road_div[:, :2]), fill=False, color="orange", linestyle="-"
                    )
                    # linewidth=2)
                    plt.gca().add_patch(linepatch)
            elif layer_k in {"road_boundaries"}:
                for road_bound in layer_v:
                    linepatch = PathPatch(
                        Path(road_bound.polyline[:, :2]),
                        fill=False,
                        color="darkgray",
                        linestyle="-",
                    )
                    # linewidth=2)
                    plt.gca().add_patch(linepatch)
            # elif layer_k in {'lane_channels'}:
            #     for lane_channel in layer_v:
            #         linepatch = PathPatch(Path(lane_channel.left),
            #                             fill=False,
            #                             color='blue',
            #                             linestyle='-')
            #                             # linewidth=2)
            #         plt.gca().add_patch(linepatch)
            #         linepatch = PathPatch(Path(lane_channel.right),
            #                             fill=False,
            #                             color='red',
            #                             linestyle='-')
            #                             # linewidth=2)
            #         plt.gca().add_patch(linepatch)

        # plot local coordinate system origin
        if poses is None:
            local_coords = np.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [arr_len, 0.0, 0.0, 1.0],
                    [0.0, arr_len, 0.0, 1.0],
                ]
            )
            world_coords = np.dot(seg.local2world, local_coords.T).T
            world_coords = world_coords[:, :2]  # only plot 2D coords
            origins.append(world_coords[0])
            xdelta = world_coords[1] - world_coords[0]
            ydelta = world_coords[2] - world_coords[0]
            plt.arrow(
                world_coords[0, 0],
                world_coords[0, 1],
                xdelta[0],
                xdelta[1],
                color="red",
            )
            plt.arrow(
                world_coords[0, 0],
                world_coords[0, 1],
                ydelta[0],
                ydelta[1],
                color="green",
            )

    if poses is not None and poses_valid is not None and poses_lwh is not None:
        # center on ego
        origin = poses[0, 0, :2, -1]
        if RIVERMARK_VIZ:
            extent = 45
            origin = origin + np.array([extent - 15.0, 0.0])

        # if RIVERMARK_VIZ and RIVERMARK_VIZ_LAYER > 3:
        #     # plot ego traj
        #     ego_pos = ego_traj[::10,:2,3]
        #     plt.plot(ego_pos[:,0], ego_pos[:,1], 'o-', c=NVCOLOR2, markersize=2.5) #, markersize=8), linewidth

        plt.xlim(origin[0] - extent, origin[0] + extent)
        plt.ylim(origin[1] - extent, origin[1] + extent)
        for n in range(poses.shape[0]):
            if RIVERMARK_VIZ and RIVERMARK_VIZ_LAYER < 1:
                continue
            if RIVERMARK_VIZ and RIVERMARK_VIZ_LAYER < 3 and n != 0:
                continue
            if RIVERMARK_VIZ and RIVERMARK_VIZ_LAYER < 4 and pose_ids[n] == "extra":
                continue
            # if RIVERMARK_VIZ and RIVERMARK_VIZ_LAYER == 3 and n == 0:
            #     continue
            cur_color = plt_color((n + 2) % 9)
            if RIVERMARK_VIZ and RIVERMARK_VIZ_LAYER >= 4 and pose_ids[n] == "extra":
                cur_color = "#ff00ff"
            if RIVERMARK_VIZ:
                print(n)
                print(pose_ids[n])
                print(cur_color)
            cur_poses = poses[n]  # , ::20]
            xy = cur_poses[:, :2, 3]
            hvec = cur_poses[:, :2, 0]  # x axis
            hvec = hvec / np.linalg.norm(hvec, axis=1, keepdims=True)
            for t in range(cur_poses.shape[0]):
                if poses_valid[n, t]:
                    plot_box(
                        np.array([xy[t, 0], xy[t, 1], hvec[t, 0], hvec[t, 1]]),
                        poses_lwh[n, :2],
                        color=NVCOLOR if n == 0 else cur_color,
                        alpha=1.0,
                        no_heading=False,
                    )
                    if pose_ids is not None and not RIVERMARK_VIZ:
                        plt.text(
                            xy[t, 0] + 1.0,
                            xy[t, 1] + 1.0,
                            pose_ids[n],
                            c="red",
                            fontsize="x-small",
                        )

    plt.gca().set_aspect("equal")
    plt.grid(grid)
    if not show_ticks:
        plt.xticks([])
        plt.yticks([])
        plt.gca().axis("off")
    # plt.tight_layout()
    cur_save_path = os.path.join(comp_out_path, prefix + ".jpg")
    fig.savefig(cur_save_path)
    # plt.show()
    plt.close(fig)

    if RIVERMARK_VIZ:
        og_img = cv2.imread(cur_save_path)
        rot_img = cv2.rotate(og_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(cur_save_path, rot_img)


def get_tile_mask(tile, layer_name, local_box, canvas_size):
    """
    Rasterizes a layer of the given tile object into a binary mask.
    Assumes tile object has been converted to hold nuscenes-like layers.

    :param tile: NVMapTile object holding the nuscenes layers
    :param layer_name str: which layer to rasterize, currently supports ['drivable_area', 'carpark_area', 'road_divider', 'lane_divider']
    :param local_box tuple: (center_x, center_y, H, W) in meters which patch of the map to rasterize
    :param canvas_size tuple: (H, W) pixels tuple which determines the resolution at which the layer is rasterized
    """
    # must transform each map element to pixel space
    # https://github.com/nutonomy/nuscenes-devkit/blob/9b209638ef3dee6d0cdc5ac700c493747f5b35fe/python-sdk/nuscenes/map_expansion/map_api.py#L1894
    patch_x, patch_y, patch_h, patch_w = local_box
    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w
    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0
    trans = np.array([[trans_x, trans_y]])
    scale = np.array([[scale_width, scale_height]])

    map_mask = np.zeros(canvas_size, np.uint8)
    for seg_id, seg in tile.segments.items():
        for poly_pts in seg.layers[layer_name]:
            # convert to pixel coords
            poly_pts = (poly_pts + trans) * scale
            # rasterize
            if layer_name in {"drivable_area"}:
                # polygon
                coords = poly_pts.astype(np.int32)
                cv2.fillPoly(map_mask, [coords], 1)
            elif layer_name in {"lane_divider", "road_divider"}:
                # polyline
                coords = poly_pts.astype(np.int32)
                cv2.polylines(map_mask, [coords], False, 1, 2)
            elif layer_name in {"carpark_area"}:
                # empty
                pass
            else:
                print("Unrecognized layer %d - cannot render mask" % (layer_name))

    return map_mask


# https://www.dmv.ca.gov/portal/handbook/california-driver-handbook/lane-control/
#   - solid yellow lines = center of road for two-way (road divider)
#   - two, one solid one broken yellow = may pass but going opposite dir (road divider)
#   - two solid yellow = road divider
#   - solid white = edge of road going same way (lane divider)
#   - broken white = two or more lanes same direction (lane divider)
#   - double white = HOV (lane divider)
#   - invisible should be ommitted (e.g. will just be in intersections)


def convert_tile_to_nuscenes(tile):
    """
    Given a tile, converts its layers into similar format as nuscenes.
    This includes converting lane dividers and road boundaries to lane/road dividers and
    drivable area.
    returns an updated copy of the tile.
    """
    print("Converting to nuscenes...")
    tile = copy.deepcopy(tile)
    for seg_id, seg in tile.segments.items():
        if "lane_dividers" in seg.layers:
            nusc_lane_dividers = []
            nusc_road_dividers = []
            for div in seg.layers["lane_dividers"]:
                style = div.style
                if len(style) > 0:
                    div_color = style[0][2]
                    div_pattern = style[0][0]
                    if div_color in {"White"}:  # , 'Green'}:
                        # divides traffic in same direction
                        nusc_lane_dividers.append(div.polyline[:, :2])
                    elif div_color in {"Yellow"} or (
                        len(style) == 2 and div_pattern == "Botts Dots"
                    ):  # , 'Blue', 'Red', 'Orange'}:
                        # divides traffic in opposite direction
                        nusc_road_dividers.append(div.polyline[:, :2])
                # elif RIVERMARK_VIZ:
                #     nusc_lane_dividers.append(div.polyline[:,:2])
            # update segment
            seg.layers["lane_divider"] = nusc_lane_dividers
            seg.layers["road_divider"] = nusc_road_dividers

            del seg.layers["lane_dividers"]

        if "road_boundaries" in seg.layers:
            del seg.layers["road_boundaries"]
            pass  # actually don't need this for now, it's just for reference

        if "lane_channels" in seg.layers:
            # convert to drivable area polygon
            expand_len = NUSC_EXPAND_LEN  # meters
            drivable_area_polys = []
            for channel in seg.layers["lane_channels"]:
                left = channel.left[:, :2]
                right = channel.right[:, :2]
                if RIVERMARK_VIZ:
                    seg.layers["lane_divider"].append(left)
                    seg.layers["road_divider"].append(right)
                if left.shape[0] > 1:
                    # compute normals at each vertex to expand
                    left_diff = left[1:] - left[:-1]
                    left_diff = np.concatenate([left_diff, left_diff[-1:, :]], axis=0)
                    left_diff = left_diff / np.linalg.norm(
                        left_diff, axis=1, keepdims=True
                    )
                    left_norm = np.concatenate(
                        [-left_diff[:, 1:2], left_diff[:, 0:1]], axis=1
                    )
                    right_diff = right[1:] - right[:-1]
                    right_diff = np.concatenate(
                        [right_diff, right_diff[-1:, :]], axis=0
                    )
                    right_diff = right_diff / np.linalg.norm(
                        right_diff, axis=1, keepdims=True
                    )
                    right_norm = np.concatenate(
                        [right_diff[:, 1:2], -right_diff[:, 0:1]], axis=1
                    )
                    # expand channel
                    left = left + (left_norm * expand_len)
                    right = right + (right_norm * expand_len)

                channel_poly = np.concatenate([right, np.flip(left, axis=0)], axis=0)
                drivable_area_polys.append(channel_poly)
            seg.layers["drivable_area"] = drivable_area_polys
            del seg.layers["lane_channels"]

        # add empty carpark area for completeness
        seg.layers["carpark_area"] = []

    # compute extents of map
    map_maxes = np.array([-float("inf"), -float("inf")])  # xmax, ymax
    map_mins = np.array([float("inf"), float("inf")])  # xmin, ymin
    for _, seg in tile.segments.items():
        for k, v in seg.layers.items():
            if len(v) > 0:
                all_pts = np.concatenate(v, axis=0)
                cur_maxes = np.amax(all_pts, axis=0)
                cur_mins = np.amin(all_pts, axis=0)
                map_maxes = np.where(cur_maxes > map_maxes, cur_maxes, map_maxes)
                map_mins = np.where(cur_mins < map_mins, cur_mins, map_mins)
    map_xlim = (map_mins[0] - 10, map_maxes[0] + 10)  # buffer of 10m
    map_ylim = (map_mins[1] - 10, map_maxes[1] + 10)  # buffer of 10m
    W = map_xlim[1] - map_xlim[0]
    H = map_ylim[1] - map_ylim[0]
    # translate so bottom left corner is at origin
    trans_offset = np.array([[-map_xlim[0], -map_ylim[0]]])
    tile.trans_offset = trans_offset
    tile.H = H
    tile.W = W
    for _, seg in tile.segments.items():
        seg.local2world[:2, -1] += trans_offset[0]
        for k, v in seg.layers.items():
            if len(v) > 0:
                for pts in v:
                    pts += trans_offset

    return tile


def load_tile(tile_path, layers=["lane_dividers_v1", "lane_channels_v1"]):
    # load in all road segment dicts
    print("Parsing segment graph...")
    tile_name = tile_path.split("/")[-1]
    segs_path = os.path.join(tile_path, SEGMENT_GRAPH_LAYER)
    assert os.path.exists(
        tile_path
    ), "cannot find segment graph layer, which is required to load any layers"
    seg_json_list, _ = load_json_dir(segs_path)
    road_segments = parse_road_segments(seg_json_list)
    print("Found %d road segments:" % (len(road_segments)))

    # fill road segments with other desired layers
    print("Loading requested layers...")
    for layer in layers:
        assert (
            layer in NVMAP_LAYERS
        ), "loading layer type %s is currently not supported!" % (layer)
        layer_dir = os.path.join(tile_path, layer)
        assert os.path.exists(
            layer_dir
        ), "could not find requested layer %s in tile directory!" % (layer)
        layer_json_list, seg_names = load_json_dir(layer_dir)
        if layer == "lane_dividers_v1":
            parse_lane_dividers(layer_json_list, seg_names, road_segments)
        elif layer == "lane_channels_v1":
            parse_lane_channels(layer_json_list, seg_names, road_segments)
        elif layer == "lanes_v1":
            raise NotImplementedError()

    return NVMapTile(road_segments, name=tile_name)


def load_json_dir(json_dir_path):
    """
    Loads in all json files in the given directory and returns the resulting list of dicts along
    with the names of the json files read from.
    """
    json_files = sorted(glob.glob(os.path.join(json_dir_path, "*.json")))
    file_names = [".".join(jf.split("/")[-1].split(".")[:-1]) for jf in json_files]
    json_list = []
    for jf in json_files:
        with open(jf, "r") as f:
            json_list.append(json.load(f))
    return json_list, file_names


def parse_pt(pt_dict):
    pt_entries = ["x", "y", "z", "w"]
    if len(pt_dict) == 3:
        pt = [pt_dict[pt_entries[0]], pt_dict[pt_entries[1]], pt_dict[pt_entries[2]]]
    elif len(pt_dict) == 4:
        pt = [
            pt_dict[pt_entries[0]],
            pt_dict[pt_entries[1]],
            pt_dict[pt_entries[2]],
            pt_dict[pt_entries[3]],
        ]
    else:
        assert False, "input point must be length 3 or 4"
    return pt


def parse_pt_list(pt_list):
    return np.array([parse_pt(pt) for pt in pt_list])


def parse_lane_channels(lane_channels_json_list, seg_name_list, road_segments):
    """
    Parses lane channels in each segment to an object, and store in its respective segment.
    Transforms channels into the global coordinate system.

    :param lane_channels_json_list: list of lane channels json dicts
    :param seg_name_list: the name of the segment corresponding to each json file in lane_div_json_list
    :param road_segments: dict of all road segments in a tile
    :return: updated road_segments (also updated in place)
    """
    for channel_dict, seg_name in zip(lane_channels_json_list, seg_name_list):
        cur_seg = road_segments[seg_name]
        # load lane dividers
        lane_channels = []
        lane_channel_dicts = channel_dict["channels"]
        for channel in lane_channel_dicts:
            # left
            left_geom = channel["left_side"]["geometry"][0]["chunk"][
                "channel_edge_line"
            ]
            left_polyline = parse_pt_list(left_geom["points"])
            left_polyline = cur_seg.to_global(left_polyline)[:, :3]
            # right
            right_geom = channel["right_side"]["geometry"][0]["chunk"][
                "channel_edge_line"
            ]
            right_polyline = parse_pt_list(right_geom["points"])
            right_polyline = cur_seg.to_global(right_polyline)[:, :3]
            assert (
                left_polyline.shape[0] == right_polyline.shape[0]
            ), "channel edges should be same length!"
            # build lane channel object
            lane_channels.append(LaneChannel(left_polyline, right_polyline))
        cur_seg.layers["lane_channels"] = lane_channels

    return road_segments


def parse_lane_dividers(lane_div_json_list, seg_name_list, road_segments):
    """
    Parses lane dividers in each segment to an object, and store in its respective segment.
    Transforms dividers into the global coordinate system.

    :param lane_div_json_list: list of lane divider json dicts
    :param seg_name_list: the name of the segment corresponding to each json file in lane_div_json_list
    :param road_segments: dict of all road segments in a tile
    :return: updated road_segments (also updated in place)
    """
    for div_dict, seg_name in zip(lane_div_json_list, seg_name_list):
        cur_seg = road_segments[seg_name]
        # load lane dividers
        lane_divs = []
        lane_div_dicts = div_dict["dividers"]
        for div in lane_div_dicts:
            # NOTE: left/right/height lines also sometimes available - ignoring for now
            # NOTE: lane divider styles (type/color) also available - ignoring for now
            # center_line is only guaranteed
            lane_geom = div["geometry"]["center_line"]
            lane_polyline = parse_pt_list(lane_geom["points"])
            lane_polyline = cur_seg.to_global(lane_polyline)[:, :3]
            # parse divider style
            if "style" in div:
                lane_styles = div["style"]
                lane_styles = [
                    (style["pattern"], style["material"], style["color"])
                    for style in lane_styles
                ]
            else:
                lane_styles = []
            # build lane div object
            lane_divs.append(LaneDivider(lane_polyline, lane_styles))
        cur_seg.layers["lane_dividers"] = lane_divs

        # load road boundaries
        road_bounds = []
        road_bound_dicts = div_dict["road_boundaries"]
        for div in road_bound_dicts:
            # NOTE: left/right/height lines also sometimes available - ignoring for now
            # NOTE: road boundary type also available - ignoring for now
            # center_line is only guaranteed
            bound_geom = div["geometry"]["center_line"]
            bound_polyline = parse_pt_list(bound_geom["points"])
            bound_polyline = cur_seg.to_global(bound_polyline)[:, :3]
            # parse boundary type
            if "type" in div:
                bound_type = div["type"]
            else:
                bound_type = None
            # build object
            road_bounds.append(RoadBoundary(bound_polyline, bound_type))
        cur_seg.layers["road_boundaries"] = road_bounds

    return road_segments


def parse_road_segments(seg_json_list):
    """
    Parses road segments into objects, and transforms into a shared coordinate system.

    :param seg_json_list: list of road segment json dicts

    :return: dict of all road segments mapping id -> RoadSegment
    """
    # build objects for all road segments
    road_segments = OrderedDict()
    for seg_dict in seg_json_list:
        cur_seg = build_road_seg(seg_dict)
        road_segments[cur_seg.id] = cur_seg

    # go through and annotate local2world by converting GPS to the "global" coordinate system
    for seg_id, seg in road_segments.items():
        lat_lng_alt = np.array(seg.gps_origin).reshape((-1, 3))
        rot_axis = np.array([1.0, 0.0, 0.0]).reshape((-1, 3))
        rot_angle = np.array([0.0]).reshape((-1, 1))
        ecef_pose = lat_lng_alt_2_ecef(lat_lng_alt, rot_axis, rot_angle, "WGS84")[0]
        seg.local2world = np.linalg.inv(GLOBAL_BASE_POSE) @ ecef_pose

    return road_segments


def build_road_seg(seg_dict):
    """
    Parses road segment json dictionary into an object.
    """
    segment = seg_dict["segment"]
    seg_id = segment["id"]
    seg_origin = [
        segment["origin"]["lat"],
        segment["origin"]["lon"],
        segment["origin"]["height"],
    ]
    connections = []
    if "connections" in segment:
        conn_list = segment["connections"]
        for conn in conn_list:
            source_id = conn["source_id"]
            source2tgt = []
            for ci in range(4):
                source2tgt.append(
                    np.array(parse_pt(conn["source_to_target"][f"column_{ci}"]))
                )
            source2tgt = np.stack(source2tgt, axis=1)
            connections.append((source_id, source2tgt))
    return RoadSegment(seg_id, seg_origin, connections)


def collect_seg_origins(road_segments):
    """
    Returns np array of all road_segment origins (in order of dict).
    :param road_segments: OrderedDict of RoadSegment objects
    """
    return np.array([seg.origin for _, seg in road_segments.items()])


class RoadSegment(object):
    def __init__(self, seg_id, origin, connections, is_root=False, layers=None):
        """
        :param seg_id str:
        :param origin: list of [lat, lon, height]
        :param connections list: list of tuples, each containing (neighbor_id, neighbor2local_transform)
                            where the transform is a np.array(4,4))
        :param layers dict: layer objects within this road segment
        """
        self.id = seg_id
        self.gps_origin = origin  # GPS
        self.connections = connections
        self.layers = layers if layers is not None else dict()
        # transformation matrix from local to world frame
        self.local2world = None

    def to_global(self, pts):
        """
        Transform an array of points from this segment's frame to global.

        :param pts: np array (N x 3)
        """
        pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        pts = np.dot(self.local2world, pts.T).T
        return pts

    def transform(self, mat):
        """
        Transforms this segment and all contained layers by the given 4x4 transformation matrix.
        """
        self.local2world = mat @ self.local2world
        for _, layer in self.layers.items():
            for el in layer:
                el.transform(mat)

    def __repr__(self):
        return "<RoadSegment -- %s>" % str(
            "\n ".join("%s : %s" % (k, repr(v)) for (k, v) in self.__dict__.items())
        )


class LaneDivider(object):
    def __init__(self, polyline, style):
        """
        :param polyline: np.array Nx3 defining the divider geometry
        :param style: list of tuples of (pattern, matterial, color)
        """
        self.polyline = polyline
        self.style = style

    def transform(self, mat):
        """
        Transforms this map element by the given 4x4 transformation matrix.
        """
        pts = np.concatenate(
            [self.polyline, np.ones((self.polyline.shape[0], 1))], axis=1
        )
        pts = np.dot(mat, pts.T).T
        self.polyline = pts[:, :3]

    def __repr__(self):
        return "<LaneDivider -- %s>" % str(
            "\n ".join("%s : %s" % (k, repr(v)) for (k, v) in self.__dict__.items())
        )


class RoadBoundary(object):
    def __init__(self, polyline, bound_type):
        """
        :param polyline: np.array Nx3 defining the divider geometry
        :param type str: type of the boundary
        """
        self.polyline = polyline
        self.bound_type = bound_type

    def transform(self, mat):
        """
        Transforms this map element by the given 4x4 transformation matrix.
        """
        pts = np.concatenate(
            [self.polyline, np.ones((self.polyline.shape[0], 1))], axis=1
        )
        pts = np.dot(mat, pts.T).T
        self.polyline = pts[:, :3]

    def __repr__(self):
        return "<RoadBoundary -- %s>" % str(
            "\n ".join("%s : %s" % (k, repr(v)) for (k, v) in self.__dict__.items())
        )


class LaneChannel(object):
    def __init__(self, left_polyline, right_polyline):
        """
        :param left_polyline: np.array Nx3 defining the channel left edge geometry
        :param right_polyline: np.array Nx3 defining the channel right edge geometry
        """
        self.left = left_polyline
        self.right = right_polyline

    def transform(self, mat):
        """
        Transforms this map element by the given 4x4 transformation matrix.
        """
        pts = np.concatenate([self.left, np.ones((self.left.shape[0], 1))], axis=1)
        pts = np.dot(mat, pts.T).T
        self.left = pts[:, :3]
        pts = np.concatenate([self.right, np.ones((self.right.shape[0], 1))], axis=1)
        pts = np.dot(mat, pts.T).T
        self.right = pts[:, :3]

    def __repr__(self):
        return "<LaneChannel -- %s>" % str(
            "\n ".join("%s : %s" % (k, repr(v)) for (k, v) in self.__dict__.items())
        )


class NVMapTile(object):
    def __init__(self, segments, name=None):
        """
        :param segments: dict mapping seg_id -> RoadSegment objects for all road segments in the tile.
        """
        self.segments = segments
        self.name = name

    def transform(self, mat):
        """
        Multiply all elements of the map tile by the given 4x4 matrix
        """
        for seg_id, seg in self.segments.items():
            seg.transform(mat)

    def __repr__(self):
        return "<NVMapTile -- %s>" % str(
            "\n ".join("%s : %s" % (k, repr(v)) for (k, v) in self.__dict__.items())
        )


#######################################################################################################################

#
#  Utils from Zan for converting between GPS and world coordinates
#

from scipy.spatial.transform import Rotation as R


def lat_lng_alt_2_ecef(
    lat_lng_alt, orientation_axis, orientation_angle, earth_model="WGS84"
):
    """Computes the transformation from the world pose coordiante system to the earth centered earth fixed (ECEF) one
    Args:
        lat_lng_alt (np.array): latitude, longitude and altitude coordinate (in degrees and meters) [n,3]
        orientation_axis (np.array): orientation in the local ENU coordinate system [n,3]
        orientation_angle (np.array): orientation angle of the local ENU coordinate system in degrees [n,1]
        earth_model (string): earth model used for conversion (spheric will be unaccurate when maps are large)
    Out:
        trans (np.array): transformation parameters from world pose to ECEF coordinate system in se3 form (n, 4, 4)
    """
    n = lat_lng_alt.shape[0]
    trans = np.tile(np.eye(4).reshape(1, 4, 4), [n, 1, 1])

    theta = (90.0 - lat_lng_alt[:, 0]) * np.pi / 180
    phi = lat_lng_alt[:, 1] * np.pi / 180

    R_enu_ecef = local_ENU_2_ECEF_orientation(theta, phi)

    if earth_model == "WGS84":
        a = 6378137.0
        flattening = 1.0 / 298.257223563
        b = a * (1.0 - flattening)
        translation = lat_lng_alt_2_translation_ellipsoidal(lat_lng_alt, a, b)

    elif earth_model == "sphere":
        earth_radius = 6378137.0  # Earth radius in meters
        z_dir = np.concatenate(
            [
                (np.sin(theta) * np.cos(phi))[:, None],
                (np.sin(theta) * np.sin(phi))[:, None],
                (np.cos(theta))[:, None],
            ],
            axis=1,
        )

        translation = (earth_radius + lat_lng_alt[:, -1])[:, None] * z_dir

    else:
        raise ValueError("Selected ellipsoid not implemented!")

    world_pose_orientation = axis_angle_2_so3(orientation_axis, orientation_angle)

    trans[:, :3, :3] = R_enu_ecef @ world_pose_orientation
    trans[:, :3, 3] = translation

    return trans


def local_ENU_2_ECEF_orientation(theta, phi):
    """Computes the rotation matrix between the world_pose and ECEF coordinate system
    Args:
        theta (np.array): theta coordinates in radians [n,1]
        phi (np.array): phi coordinates in radians [n,1]
    Out:
        (np.array): rotation from world pose to ECEF in so3 representation [n,3,3]
    """
    z_dir = np.concatenate(
        [
            (np.sin(theta) * np.cos(phi))[:, None],
            (np.sin(theta) * np.sin(phi))[:, None],
            (np.cos(theta))[:, None],
        ],
        axis=1,
    )
    z_dir = z_dir / np.linalg.norm(z_dir, axis=-1, keepdims=True)

    y_dir = np.concatenate(
        [
            -(np.cos(theta) * np.cos(phi))[:, None],
            -(np.cos(theta) * np.sin(phi))[:, None],
            (np.sin(theta))[:, None],
        ],
        axis=1,
    )
    y_dir = y_dir / np.linalg.norm(y_dir, axis=-1, keepdims=True)

    x_dir = np.cross(y_dir, z_dir)

    return np.concatenate(
        [x_dir[:, :, None], y_dir[:, :, None], z_dir[:, :, None]], axis=-1
    )


def lat_lng_alt_2_translation_ellipsoidal(lat_lng_alt, a, b):
    """Computes the translation based on the ellipsoidal earth model
    Args:
        lat_lng_alt (np.array): latitude, longitude and altitude coordinate (in degrees and meters) [n,3]
        a (float/double): Semi-major axis of the ellipsoid
        b (float/double): Semi-minor axis of the ellipsoid
    Out:
        (np.array): translation from world pose to ECEF [n,3]
    """

    phi = lat_lng_alt[:, 0] * np.pi / 180
    gamma = lat_lng_alt[:, 1] * np.pi / 180

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    e_square = (a * a - b * b) / (a * a)

    N = a / np.sqrt(1 - e_square * sin_phi * sin_phi)

    x = (N + lat_lng_alt[:, 2]) * cos_phi * cos_gamma
    y = (N + lat_lng_alt[:, 2]) * cos_phi * sin_gamma
    z = (N * (b * b) / (a * a) + lat_lng_alt[:, 2]) * sin_phi

    return np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)


def axis_angle_2_so3(axis, angle, degrees=True):
    """Converts the axis angle representation of the so3 rotation matrix
    Args:
        axis (np.array): the rotation axes [n,3]
        angle float/double: rotation angles either in degrees or radians [n]
        degrees bool: True if angle is given in degrees else False

    Out:
        (np array): rotations given so3 matrix representation [n,3,3]
    """
    # Treat angle (radians) below this as 0.
    cutoff_angle = 1e-9 if not degrees else 1e-9 * 180 / np.pi
    angle[angle < cutoff_angle] = 0.0

    # Scale the axis to have the norm representing the angle
    if degrees:
        angle = np.radians(angle)
    axis_angle = (angle / np.linalg.norm(axis, axis=1, keepdims=True)) * axis

    return R.from_rotvec(axis_angle).as_matrix()


def ecef_2_lat_lng_alt(trans, earth_model="WGS84"):
    """Converts the transformation from the earth centered earth fixed (ECEF) coordinate frame to the world pose
    Args:
        trans (np.array): transformation parameters in ECEF [n,4,4]
        earth_model (string): earth model used for conversion (spheric will be unaccurate when maps are large)
    Out:
        lat_lng_alt (np.array): latitude, longitude and altitude coordinate (in degrees and meters) [n,3]
        orientation_axis (np.array): orientation in the local ENU coordinate system [n,3]
        orientation_angle (np.array): orientation angle of the local ENU coordinate system in degrees [n,1]
    """

    translation = trans[:, :3, 3]
    rotation = trans[:, :3, :3]

    if earth_model == "WGS84":
        a = 6378137.0
        flattening = 1.0 / 298.257223563
        lat_lng_alt = translation_2_lat_lng_alt_ellipsoidal(translation, a, flattening)

    elif earth_model == "sphere":
        earth_radius = 6378137.0  # Earth radius in meters
        lat_lng_alt = translation_2_lat_lng_alt_spherical(translation, earth_radius)

    else:
        raise ValueError("Selected ellipsoid not implemented!")

    # Compute the orientation axis and angle
    theta = (90.0 - lat_lng_alt[:, 0]) * np.pi / 180
    phi = lat_lng_alt[:, 1] * np.pi / 180

    R_ecef_enu = local_ENU_2_ECEF_orientation(theta, phi).transpose(0, 2, 1)

    orientation = R_ecef_enu @ rotation
    orientation_axis, orientation_angle = so3_2_axis_angle(orientation)

    return lat_lng_alt, orientation_axis, orientation_angle


def translation_2_lat_lng_alt_spherical(translation, earth_radius):
    """Computes the translation in the ECEF to latitude, longitude, altitude based on the spherical earth model
    Args:
        translation (np.array): translation in the ECEF coordinate frame (in meters) [n,3]
        earth_radius (float/double): earth radius
    Out:
        (np.array): latitude, longitude and altitude [n,3]
    """
    altitude = np.linalg.norm(translation, axis=-1) - earth_radius
    latitude = (
        90
        - np.arccos(
            translation[:, 2] / np.linalg.norm(translation, axis=-1, keepdims=True)
        )
        * 180
        / np.pi
    )
    longitude = np.arctan2(translation[:, 1], translation[:, 0]) * 180 / np.pi

    return np.concatenate(
        [latitude[:, None], longitude[:, None], altitude[:, None]], axis=1
    )


def translation_2_lat_lng_alt_ellipsoidal(translation, a, f):
    """Computes the translation in the ECEF to latitude, longitude, altitude based on the ellipsoidal earth model
       Args:
           translation (np.array): translation in the ECEF coordinate frame (in meters) [n,3]
           a (float/double): Semi-major axis of the ellipsoid
           f (float/double): flattening factor of the earth
    radius
       Out:
           (np.array): latitude, longitude and altitude [n,3]
    """

    # Compute support parameters
    f0 = (1 - f) * (1 - f)
    f1 = 1 - f0
    f2 = 1 / f0 - 1

    z_div_1_f = translation[:, 2] / (1 - f)
    x2y2 = np.square(translation[:, 0]) + np.square(translation[:, 1])

    x2y2z2 = x2y2 + z_div_1_f * z_div_1_f
    x2y2z2_pow_3_2 = x2y2z2 * np.sqrt(x2y2z2)

    gamma = (
        (x2y2z2_pow_3_2 + a * f2 * z_div_1_f * z_div_1_f)
        / (x2y2z2_pow_3_2 - a * f1 * x2y2)
        * translation[:, 2]
        / np.sqrt(x2y2)
    )

    longitude = np.arctan2(translation[:, 1], translation[:, 0]) * 180 / np.pi
    latitude = np.arctan(gamma) * 180 / np.pi
    altitude = np.sqrt(1 + np.square(gamma)) * (
        np.sqrt(x2y2) - a / np.sqrt(1 + f0 * np.square(gamma))
    )

    return np.concatenate(
        [latitude[:, None], longitude[:, None], altitude[:, None]], axis=1
    )


def so3_2_axis_angle(so3, degrees=True):
    """Converts the so3 representation to axis_angle
    Args:
        so3 (np.array): the rotation matrices [n,3,3]
        degrees bool: True if angle should be given in degrees

    Out:
        axis (np array): the rotation axis [n,3]
        angle (np array): the rotation angles, either in degrees (if degrees=True) or radians [n,]
    """
    rot_vec = R.from_matrix(so3).as_rotvec()

    angle = np.linalg.norm(rot_vec, axis=-1, keepdims=True)
    axis = rot_vec / angle
    if degrees:
        angle = np.degrees(angle)

    return axis, angle


#######################################################################################################################

#
# Utils for loading in ego and autolabel pose data for session
#

import datetime
import pickle

from scipy import interpolate, spatial

# NV_EGO_LWH = [4.084, 1.73, 1.562] # this is the nuscenes measurements
NV_EGO_LWH = [5.30119, 2.1133, 1.49455]  # actual hyperion 8


class PoseInterpolator:
    """Interpolates the poses to the desired time stamps. The translation component is interpolated linearly,
    while spherical linear interpolation (SLERP) is used for the rotations.
    https://en.wikipedia.org/wiki/Slerp

    Args:
        poses (np.array): poses at given timestamps in a se3 representation [n,4,4]
        timestamps (np.array): timestamps of the known poses [n]
        ts_target (np.array): timestamps for which the poses will be interpolated [m,1]
    Out:
        (np.array): interpolated poses in se3 representation [m,4,4]
    """

    def __init__(self, poses, timestamps):

        self.slerp = spatial.transform.Slerp(
            timestamps, R.from_matrix(poses[:, :3, :3])
        )
        self.f_x = interpolate.interp1d(timestamps, poses[:, 0, 3])
        self.f_y = interpolate.interp1d(timestamps, poses[:, 1, 3])
        self.f_z = interpolate.interp1d(timestamps, poses[:, 2, 3])

        self.last_row = np.array([0, 0, 0, 1]).reshape(1, 1, -1)

    def interpolate_to_timestamps(self, ts_target):
        x_interp = self.f_x(ts_target).reshape(-1, 1, 1)
        y_interp = self.f_y(ts_target).reshape(-1, 1, 1)
        z_interp = self.f_z(ts_target).reshape(-1, 1, 1)
        R_interp = self.slerp(ts_target).as_matrix().reshape(-1, 3, 3)

        t_interp = np.concatenate([x_interp, y_interp, z_interp], axis=-2)

        return np.concatenate(
            (
                np.concatenate([R_interp, t_interp], axis=-1),
                np.tile(self.last_row, (R_interp.shape[0], 1, 1)),
            ),
            axis=1,
        )


def angle_diff(x, y, period=2 * np.pi):
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: angle 1 (B)
    :param y: angle 2 (B)
    :param period: periodicity in radians for assessing difference.
    :return diff: smallest angle difference between to angles (B)
    """
    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (
        2 * np.pi
    )  # shift (pi, 2*pi] to (-pi, 0]
    return diff


def load_ego_pose_from_image_meta(images_path, map_tile=None):
    """
    Loads in the SDC pose from the metadata attached to a session image stream.

    :param images_path str: directory of the images/metadata. Should contain *.pkl for each frame and timestamps.npz
    :param map_tile: Tile object, if given translates the ego trajectory so in the same frame as this map tile.
    """
    frame_meta = sorted(glob.glob(os.path.join(images_path, "*.pkl")))
    timestamps_pth = os.path.join(images_path, "timestamps.npz")

    # load in timesteps
    # ego_t = np.load(timestamps_pth)['frame_t']

    ego_poses = []
    ego_t = []
    for meta_file in frame_meta:
        with open(meta_file, "rb") as f:
            cur_meta = pickle.load(f)
            # ego_poses.append(cur_meta['ego_pose_s'])
            # ego_poses.append(cur_meta['ego_pose_timestamps'][0])
            ego_poses.append(cur_meta["ego_pose_e"])
            ego_t.append(cur_meta["ego_pose_timestamps"][1])
    ego_poses = np.stack(ego_poses, axis=0)
    ego_t = np.array(ego_t)

    # pose_sidx = int(frame_meta[0].split('/')[-1].split('.')[0])
    # pose_eidx = int(frame_meta[-1].split('/')[-1].split('.')[0]) + 1
    # ego_t = ego_t[pose_sidx:pose_eidx]

    if map_tile is not None:
        ego_poses[:, :2, -1] += map_tile.trans_offset

    return ego_poses, ego_t


def check_time_overlap(s0, e0, s1, e1):
    overlap = (
        (s0 < s1 and e0 > s1)
        or (s0 > s1 and s0 < e1)
        or (s1 < s0 and e1 > s0)
        or (s1 > s0 and s1 < e0)
    )
    return overlap


def load_trajectories(
    autolabels_path,
    ego_images_path,
    lidar_rig_path,
    map_tile=None,
    frame_range=None,
    postprocess=True,
    fill_first_n=None,
    mine_dups=False,
    extra_obj_path=None,
    load_ids=None,
    crop2valid=True,
):
    """
    This only loads labeled trajectories that are available at the same section
    as the ego labels.

    :param autolabels_path str: pkl file to load autolabels from
    :param ego_images_path str: directory containing image metadata to load ego poses from
    :param lidar_rig_path str: npz containing lidar2rig transform for ego
    :param map_tile: Tile object, if given translates the trajectories so in the same frame as this map tile.
    :param frame_range tuple: If given (start, end) only loads in this frame range (wrt the ego sequence)
    :param postprocess bool: If true, runs some post-processing to associate track and heuristically remove rotation flips.
    :param extra_obj_path str: if given, load in an additional trajectory from here and includes in the data
    :return traj_T:
    """
    # Load the autolabels
    with open(autolabels_path, "rb") as f:
        labels = pickle.load(f)

    # Load the poses and the timestamps
    ego_poses, ego_pose_timestamps = load_ego_pose_from_image_meta(ego_images_path)
    if frame_range is not None:
        assert frame_range[1] > frame_range[0]
        assert frame_range[0] >= 0 and frame_range[0] <= ego_poses.shape[0]
        ego_poses = ego_poses[frame_range[0] : frame_range[1]]
        ego_pose_timestamps = ego_pose_timestamps[frame_range[0] : frame_range[1]]

    # if ego_poses are not in rivermark frame, need to take them so can load autolabels
    ego_poses_rivermark = ego_poses
    if GLOBAL_BASE_POSE is not GLOBAL_BASE_POSE_RIVERMARK:
        ego_poses_ecef = np.matmul(GLOBAL_BASE_POSE[np.newaxis], ego_poses)
        ego_poses_rivermark = np.matmul(
            np.linalg.inv(GLOBAL_BASE_POSE_RIVERMARK)[np.newaxis], ego_poses_ecef
        )

    # Load the lidar to rig transformation parameters and timestamps
    T_lidar_rig = np.load(lidar_rig_path)["T_lidar_rig"]

    # first pass to break tracks into contiguous subsequences
    #       and merge manually given missed associations
    track_seqs = dict()
    processed_ids = set()
    updated_labels = dict()
    for track_id, track in labels.items():
        if load_ids is not None and track_id not in load_ids:
            continue
        if track_id in processed_ids or track_id in TRACK_REMOVE:
            # already processed this through association
            #       or should be removed
            continue

        obj_ts = track["3D_bbox"][:, 0]
        if track_id in TRACK_ASSOC:
            # stack all the data from all associated tracks
            # NOTE: this assumes TRACK_ASSOC is sorted in temporal order already
            assoc_data = [
                labels[assoc_track_id]["3D_bbox"]
                for assoc_track_id in TRACK_ASSOC[track_id]
            ]
            # if association is wrong, the tracks may overlap
            valid_assoc = [
                not check_time_overlap(
                    obj_ts[0], obj_ts[-1], assoc_label[0, 0], assoc_label[-1, 0]
                )
                for assoc_label in assoc_data
            ]
            if np.sum(valid_assoc) != len(valid_assoc):
                print("Invalid associations for track_id %s!!" % (track_id))
                print("Ignoring: ")
                print(np.array(TRACK_ASSOC[track_id])[~np.array(valid_assoc)])
            assoc_data = [
                assoc_label
                for aid, assoc_label in enumerate(assoc_data)
                if valid_assoc[aid]
            ]
            if len(assoc_data) > 0:
                assoc_bbox = np.concatenate([track["3D_bbox"]] + assoc_data, axis=0)
                updated_labels[track_id] = {
                    "3D_bbox": assoc_bbox,
                    "type": track["type"],
                }
                obj_ts = assoc_bbox[:, 0]
                processed_ids.update(TRACK_ASSOC[track_id])
            else:
                updated_labels[track_id] = track
        else:
            updated_labels[track_id] = track

        if len(obj_ts) < 2:
            # make sure track is longer than single frame
            continue
        track_seqs[track_id] = []
        # larger than 3 timesteps considered a break, otherwise should be reasonable to interpolate
        #       should we do even larger?
        track_break = np.diff(1e-6 * obj_ts) > (AUTOLABEL_DT * 3 + AUTOLABEL_DT * 0.5)
        seq_sidx = 0
        for tidx in range(1, obj_ts.shape[0]):
            if track_break[tidx - 1]:
                track_seqs[track_id].append((seq_sidx, tidx))
                seq_sidx = tidx
        track_seqs[track_id].append((seq_sidx, obj_ts.shape[0]))
        processed_ids.add(track_id)

    # load extra object
    if extra_obj_path is not None:
        extra_obj_data = np.load(extra_obj_path)
        extra_obj_poses = extra_obj_data["poses"]
        extra_obj_timestamps = extra_obj_data["pose_timestamps"]
        extra_obj_lwh = EXTRA_DYN_LWH
        extra_track = {
            "poses": extra_obj_poses,
            "timestamps": extra_obj_timestamps,
            "lwh": extra_obj_lwh,
            "type": "car",
        }
        updated_labels["extra"] = extra_track
        track_seqs["extra"] = [(0, extra_obj_poses.shape[0])]

    # collect all tracks that overlap with ego data
    traj_poses = []
    traj_valid = []
    traj_lwh = []
    traj_ids = []
    for track_id, cur_track_seqs in track_seqs.items():
        track = updated_labels[track_id]
        if track_id == "extra":
            all_obj_ts = track["timestamps"]
            all_obj_dat = track["poses"]
            obj_lwh = track["lwh"]
        else:
            all_obj_ts = track["3D_bbox"][:, 0]
            all_obj_dat = track["3D_bbox"]
            obj_lwh = np.median(
                all_obj_dat[:, 4:7], axis=0
            )  # use all timesteps for bbox size
        # will fill these in as we go through each subseq
        full_obj_traj = np.ones_like(ego_poses_rivermark) * np.nan
        obj_valid = np.zeros((full_obj_traj.shape[0]), dtype=bool)
        for seq_sidx, seq_eidx in cur_track_seqs:
            obj_ts = all_obj_ts[seq_sidx:seq_eidx]
            if (
                (
                    obj_ts[0] >= ego_pose_timestamps[0]
                    and obj_ts[0] <= ego_pose_timestamps[-1]
                )
                or (
                    obj_ts[-1] >= ego_pose_timestamps[0]
                    and obj_ts[-1] <= ego_pose_timestamps[-1]
                )
                or (
                    obj_ts[0] <= ego_pose_timestamps[0]
                    and obj_ts[-1] >= ego_pose_timestamps[-1]
                )
            ):
                obj_type = track["type"]
                # if obj_type != 'car':
                #     continue
                obj_dat = all_obj_dat[seq_sidx:seq_eidx]
                # find steps overlapping the ego sequence
                valid_ts = np.logical_and(
                    obj_ts >= ego_pose_timestamps[0], obj_ts <= ego_pose_timestamps[-1]
                )

                overlap_inds = np.nonzero(valid_ts)[0]
                if len(overlap_inds) < 2:
                    continue  # need more than 1 frame overlap
                sidx = np.amin(overlap_inds)
                eidx = np.amax(overlap_inds) + 1

                obj_ts = obj_ts[sidx:eidx]
                # some poses have the same timestep -- drop these so we can interpolate
                valid_t = np.diff(obj_ts) > 0
                valid_t = np.append(valid_t, [True])
                if not valid_t[0]:
                    # want to keep the edge times in tact since these surround ego times
                    valid_t[0] = True
                    valid_t[1] = False
                obj_ts = obj_ts[valid_t]

                if track_id == "extra":
                    print(obj_ts)
                    glob_obj_poses = obj_dat[sidx:eidx]
                    print(glob_obj_poses.shape)
                    # exit()
                else:
                    obj_pos = obj_dat[sidx:eidx, 1:4][valid_t]
                    # print(obj_pos)
                    # print(obj_dat[sidx:eidx,4:7][valid_t])
                    obj_rot_eulxyz = obj_dat[sidx:eidx, 7:][valid_t]
                    obj_rot_eulxyz[obj_rot_eulxyz[:, 2] < -np.pi, 2] += 2 * np.pi
                    obj_rot_eulxyz[obj_rot_eulxyz[:, 2] > np.pi, 2] -= 2 * np.pi
                    obj_R = R.from_euler(
                        "xyz", obj_rot_eulxyz, degrees=False
                    ).as_matrix()
                    # build transformation matrix (pose sequence)
                    obj_poses = np.repeat(np.eye(4)[np.newaxis], len(obj_ts), axis=0)
                    obj_poses[:, :3, :3] = obj_R
                    obj_poses[:, :3, -1] = obj_pos

                    # need to interpolate the ego pose to transform from lidar frame to global
                    overlap_ego_mask = np.logical_and(
                        ego_pose_timestamps >= obj_ts[0] - 1e6,
                        ego_pose_timestamps <= obj_ts[-1] + 1e6,
                    )  # add 1 sec around so can interp first/last frames
                    overlap_ego_t = ego_pose_timestamps[overlap_ego_mask]
                    overlap_ego_poses = ego_poses_rivermark[overlap_ego_mask]
                    ego_interp = PoseInterpolator(overlap_ego_poses, overlap_ego_t)
                    T_rig_global = ego_interp.interpolate_to_timestamps(obj_ts)

                    # transform to rig frame from lidar
                    rig_obj_poses = np.matmul(T_lidar_rig[np.newaxis], obj_poses)

                    # print('elev')
                    # print(rig_obj_poses[:,2, 3])
                    # print('height')
                    # print(obj_dat[sidx:eidx,6][valid_t])

                    # transform to global frame (w.r.t rivermark) from rig
                    glob_obj_poses = np.matmul(T_rig_global, rig_obj_poses)
                    # now to the desired global frame
                    if GLOBAL_BASE_POSE is not GLOBAL_BASE_POSE_RIVERMARK:
                        # to ECEF
                        glob_obj_poses = np.matmul(
                            GLOBAL_BASE_POSE_RIVERMARK[np.newaxis], glob_obj_poses
                        )
                        # to desired global pose
                        glob_obj_poses = np.matmul(
                            np.linalg.inv(GLOBAL_BASE_POSE)[np.newaxis], glob_obj_poses
                        )

                if postprocess and track_id != "extra":
                    # we're going to collect frames with "correct" orientations
                    #       by first looking at dynamic frames where can use motion to infer correct orientation,
                    #       then using dynamic to determine correctness of static frames.
                    #       then we can interpolate between all these correct frames.
                    glob_hvec = glob_obj_poses[:, :2, 0]  # x-axis
                    glob_hvec = glob_hvec / np.linalg.norm(
                        glob_hvec, axis=-1, keepdims=True
                    )
                    glob_yaw = np.arctan2(glob_hvec[:, 1], glob_hvec[:, 0])
                    glob_pos = glob_obj_poses[:, :2, 3]  # 2d

                    # TODO add smoothing to the position to avoid noisy velocities
                    obj_vel = (
                        np.diff(glob_pos[:, :2], axis=0)
                        / np.diff(obj_ts * 1e-6)[:, np.newaxis]
                    )
                    obj_vel = np.concatenate([obj_vel, obj_vel[-1:, :]], axis=0)
                    # is_dynamic = np.median(np.linalg.norm(obj_vel, axis=1)) > 2.0 # m/s
                    is_dynamic = np.linalg.norm(obj_vel, axis=1) > 2.0  # m/s

                    is_correct_mask = np.zeros((glob_pos.shape[0]), dtype=bool)

                    # dynamic first
                    if np.sum(is_dynamic) > 0:
                        dynamic_vel = obj_vel[is_dynamic]
                        dynamic_yaw = glob_yaw[is_dynamic]
                        dynamic_vel_norm = np.linalg.norm(
                            dynamic_vel, axis=1, keepdims=True
                        )
                        vel_dir = dynamic_vel / (dynamic_vel_norm + 1e-9)
                        head_dir = np.concatenate(
                            [
                                np.cos(dynamic_yaw[:, np.newaxis]),
                                np.sin(dynamic_yaw[:, np.newaxis]),
                            ],
                            axis=1,
                        )
                        vel_head_dot = np.sum(vel_dir * head_dir, axis=1)
                        dynamic_correct = vel_head_dot > 0
                        is_correct_mask[is_dynamic] = dynamic_correct

                    # now static, by referencing closest correct dynamic
                    dynamic_inds = np.nonzero(
                        np.logical_and(is_dynamic, is_correct_mask)
                    )[0]
                    static_inds = np.nonzero(~is_dynamic)[0]
                    if len(static_inds) > 0:
                        # if no dynamic frames
                        # assume correct orientation has the most frequent sign
                        num_pos = np.sum(glob_yaw[~is_dynamic] >= 0)
                        num_neg = np.sum(glob_yaw[~is_dynamic] < 0)
                        for static_ind in static_inds:
                            if len(dynamic_inds) > 0:
                                closest_dyn_ind = np.argmin(
                                    np.abs(static_ind - dynamic_inds)
                                )
                                dyn_stat_dot = np.sum(
                                    glob_hvec[closest_dyn_ind] * glob_hvec[static_ind]
                                )
                                if dyn_stat_dot > 0:  # going in same direction
                                    is_correct_mask[static_ind] = True
                            else:
                                is_wrong = (
                                    glob_yaw[static_ind] >= 0
                                    if num_neg > num_pos
                                    else glob_yaw[static_ind] < 0
                                )
                                is_correct_mask[static_ind] = not is_wrong

                    if np.sum(is_correct_mask) > 0:
                        fix_interp_poses = glob_obj_poses[is_correct_mask]
                        fix_interp_t = obj_ts[is_correct_mask]
                        # what if edges are not correct?
                        if not is_correct_mask[0]:
                            # just pad first correct to beginning
                            fix_interp_poses = np.concatenate(
                                [fix_interp_poses[0:1], fix_interp_poses], axis=0
                            )
                            fix_interp_t = np.concatenate(
                                [[obj_ts[0]], fix_interp_t], axis=0
                            )
                        if not is_correct_mask[-1]:
                            # just pad first correct to beginning
                            fix_interp_poses = np.concatenate(
                                [fix_interp_poses, fix_interp_poses[-1:]], axis=0
                            )
                            fix_interp_t = np.concatenate(
                                [fix_interp_t, [obj_ts[-1]]], axis=0
                            )

                        # now interpolate between correct frames
                        fix_flip_interp = PoseInterpolator(
                            fix_interp_poses, fix_interp_t
                        )
                        fixed_rot_poses = fix_flip_interp.interpolate_to_timestamps(
                            obj_ts
                        )
                        glob_obj_poses[:, :3, :3] = fixed_rot_poses[
                            :, :3, :3
                        ]  # don't want to update translation

                # after processing interpolate to the relevant ego timestamps (upsample from 10Hz to 30Hz)
                obj_interp = PoseInterpolator(glob_obj_poses, obj_ts)
                overlap_ego_mask = np.logical_and(
                    ego_pose_timestamps >= obj_ts[0], ego_pose_timestamps <= obj_ts[-1]
                )
                overlap_ego_t = ego_pose_timestamps[overlap_ego_mask]
                glob_obj_poses = obj_interp.interpolate_to_timestamps(overlap_ego_t)

                # update full seq information
                full_obj_traj[overlap_ego_mask] = glob_obj_poses
                obj_valid[overlap_ego_mask] = True

        if np.sum(obj_valid) > 0:
            traj_poses.append(full_obj_traj)
            traj_valid.append(obj_valid)
            traj_lwh.append(obj_lwh)
            traj_ids.append(track_id)

    traj_poses = np.stack(traj_poses, axis=0)
    traj_valid = np.stack(traj_valid, axis=0)
    traj_lwh = np.stack(traj_lwh, axis=0)
    traj_ids = np.array(traj_ids)

    if crop2valid:
        # we interpolated inside ego timestamp maximum, so have to crop a bit
        val_inds = np.nonzero(np.sum(traj_valid, axis=0) > 0)[0]
        start_valid = np.amin(val_inds)
        end_valid = np.amax(val_inds) + 1
        traj_poses = traj_poses[:, start_valid:end_valid]
        traj_valid = traj_valid[:, start_valid:end_valid]
        ego_poses = ego_poses[start_valid:end_valid]
        ego_pose_timestamps = ego_pose_timestamps[start_valid:end_valid]

        print(start_valid)
        print(end_valid)

    if fill_first_n is not None:
        # for each trajectory, make sure the first n steps are
        #       all valid either by interpolation or extrapolation.
        all_ts = ego_pose_timestamps * 1e-6
        for ai in range(traj_poses.shape[0]):
            cur_poses = traj_poses[ai]
            cur_trans = cur_poses[:, :3, 3]
            cur_R = cur_poses[:, :3, :3]
            cur_valid = traj_valid[ai]

            if np.sum(cur_valid) < 30:
                # if does't show up for at least a second throughout, don't be extrapolating
                continue

            first_n_valid = cur_valid[:fill_first_n]
            if np.sum(~first_n_valid) == 0:
                continue

            first_n_timestamps = all_ts[:fill_first_n]

            all_val_inds = sorted(np.nonzero(cur_valid)[0])
            first_val_idx = all_val_inds[0]
            last_val_idx = all_val_inds[-1]

            # interp steps are those between the first and last valid steps
            first_n_steps = np.arange(min(fill_first_n, all_ts.shape[0]))
            interp_steps = np.logical_and(
                first_n_steps >= first_val_idx, first_n_steps <= last_val_idx
            )
            first_n_interp = None
            if np.sum(interp_steps) > 0:
                first_n_interp = PoseInterpolator(
                    cur_poses[cur_valid], all_ts[cur_valid]
                )
                first_n_interp = first_n_interp.interpolate_to_timestamps(
                    first_n_timestamps[interp_steps]
                )

            # extrap fw are those past last valid step (disappear)
            extrap_fw_steps = first_n_steps > last_val_idx
            first_n_extrap_fw = None
            if np.sum(extrap_fw_steps) > 0:
                if last_val_idx > 0 and cur_valid[last_val_idx - 1]:
                    # need to compute velocity to extrapolate
                    dt = first_n_timestamps[extrap_fw_steps] - all_ts[last_val_idx]
                    dt = dt[:, np.newaxis]
                    last_pos = np.repeat(
                        cur_trans[last_val_idx][np.newaxis], dt.shape[0], axis=0
                    )
                    # translation
                    last_lin_vel = (
                        cur_trans[last_val_idx] - cur_trans[last_val_idx - 1]
                    ) / (all_ts[last_val_idx] - all_ts[last_val_idx - 1])
                    last_lin_vel = np.repeat(
                        last_lin_vel[np.newaxis], dt.shape[0], axis=0
                    )
                    extrap_trans = last_pos + last_lin_vel * dt
                    # copy rotation
                    extrap_R = np.repeat(
                        cur_R[last_val_idx : last_val_idx + 1], dt.shape[0], axis=0
                    )
                    # # extrapolate rotation
                    # last_delta_R = np.dot(cur_R[last_val_idx], cur_R[last_val_idx-1].T)
                    # last_delta_rotvec = R.from_matrix(last_delta_R).as_rotvec()
                    # last_delta_angle = np.linalg.norm(last_delta_rotvec)
                    # last_delta_axis = last_delta_rotvec / (last_delta_angle + 1e-9)
                    # last_ang_vel = last_delta_angle / (all_ts[last_val_idx] - all_ts[last_val_idx-1])
                    # last_ang_vel = np.repeat(last_ang_vel[np.newaxis,np.newaxis], dt.shape[0], axis=0)
                    # extrap_angle = last_ang_vel*dt
                    # last_delta_axis = np.repeat(last_delta_axis[np.newaxis], dt.shape[0], axis=0)
                    # extrap_rotvec = extrap_angle*last_delta_axis
                    # extrap_delta_R = R.from_rotvec(extrap_rotvec).as_matrix()
                    # extrap_R = np.matmul(extrap_delta_R, cur_R[last_val_idx:last_val_idx+1])

                    # put together
                    extrap_poses = np.repeat(np.eye(4)[np.newaxis], dt.shape[0], axis=0)
                    extrap_poses[:, :3, :3] = extrap_R
                    extrap_poses[:, :3, 3] = extrap_trans
                    first_n_extrap_fw = extrap_poses

            # extrap bw are those before first valid step (appear)
            extrap_bw_steps = first_n_steps < first_val_idx
            first_n_extrap_bw = None
            if np.sum(extrap_bw_steps) > 0:
                if (
                    first_val_idx < (cur_valid.shape[0] - 1)
                    and cur_valid[first_val_idx + 1]
                ):
                    # need to compute velocity to extrapolate
                    dt = (
                        first_n_timestamps[extrap_bw_steps] - all_ts[first_val_idx]
                    )  # note will be < 0
                    dt = dt[:, np.newaxis]
                    first_pos = np.repeat(
                        cur_trans[first_val_idx][np.newaxis], dt.shape[0], axis=0
                    )
                    # translation
                    first_lin_vel = (
                        cur_trans[first_val_idx + 1] - cur_trans[first_val_idx]
                    ) / (all_ts[first_val_idx + 1] - all_ts[first_val_idx])
                    first_lin_vel = np.repeat(
                        first_lin_vel[np.newaxis], dt.shape[0], axis=0
                    )
                    extrap_trans = first_pos + first_lin_vel * dt
                    # copy rotation
                    extrap_R = np.repeat(
                        cur_R[first_val_idx : first_val_idx + 1], dt.shape[0], axis=0
                    )
                    # # extrapolate rotation
                    # first_delta_R = np.dot(cur_R[first_val_idx+1], cur_R[first_val_idx].T)
                    # first_delta_rotvec = R.from_matrix(first_delta_R).as_rotvec()
                    # first_delta_angle = np.linalg.norm(first_delta_rotvec)
                    # first_delta_axis = first_delta_rotvec / (first_delta_angle+1e-9)
                    # first_ang_vel = first_delta_angle / (all_ts[first_val_idx+1] - all_ts[first_val_idx])
                    # first_ang_vel = np.repeat(first_ang_vel[np.newaxis,np.newaxis], dt.shape[0], axis=0)
                    # extrap_angle = first_ang_vel*dt
                    # first_delta_axis = np.repeat(first_delta_axis[np.newaxis], dt.shape[0], axis=0)
                    # extrap_rotvec = extrap_angle*first_delta_axis
                    # extrap_delta_R = R.from_rotvec(extrap_rotvec).as_matrix()
                    # extrap_R = np.matmul(extrap_delta_R, cur_R[first_val_idx:first_val_idx+1])
                    # put together
                    extrap_poses = np.repeat(np.eye(4)[np.newaxis], dt.shape[0], axis=0)
                    extrap_poses[:, :3, :3] = extrap_R
                    extrap_poses[:, :3, 3] = extrap_trans
                    first_n_extrap_bw = extrap_poses

            first_n_poses = cur_poses[:fill_first_n]
            if first_n_interp is not None:
                first_n_poses[interp_steps] = first_n_interp
            if first_n_extrap_fw is not None:
                first_n_poses[extrap_fw_steps] = first_n_extrap_fw
            if first_n_extrap_bw is not None:
                first_n_poses[extrap_bw_steps] = first_n_extrap_bw

            first_n_valid = (
                np.sum(
                    np.isnan(first_n_poses.reshape((first_n_poses.shape[0], 16))),
                    axis=1,
                )
                == 0
            )

            traj_poses[ai, :fill_first_n] = first_n_poses
            traj_valid[ai, :fill_first_n] = first_n_valid

            # if traj_ids[ai] == 2588:
            #     print(extrap_fw_steps)
            #     print(first_n_extrap_fw)
            #     exit()

        # based on fill-in can tell if there are duplicated (mis-associated tracks) if they collide
        if mine_dups:
            print("Mining possible duplicates...")
            first_n_xy = traj_poses[:, :fill_first_n, :2, 3]
            first_n_hvec = traj_poses[:, :fill_first_n, :2, 0]
            first_n_hvec = first_n_hvec / np.linalg.norm(
                first_n_hvec, axis=-1, keepdims=True
            )
            for ai in range(traj_poses.shape[0]):
                ai_mask = np.zeros((traj_poses.shape[0]), dtype=bool)
                ai_mask[ai] = True
                cur_id = traj_ids[ai]
                other_ids = traj_ids[~ai_mask]

                traj_tgt = np.concatenate([first_n_xy[ai], first_n_hvec[ai]], axis=1)
                lw_tgt = traj_lwh[ai, :2]
                traj_others = np.concatenate(
                    [first_n_xy[~ai_mask], first_n_hvec[~ai_mask]], axis=2
                )
                lw_others = traj_lwh[~ai_mask, :2]

                # if they collide more than 75% of the time
                veh_coll = check_single_veh_coll(
                    traj_tgt, lw_tgt, traj_others, lw_others
                )
                dup_mask = np.sum(veh_coll, axis=1) > int(0.75 * veh_coll.shape[1])

                if np.sum(dup_mask) > 0:
                    dup_ids = sorted(
                        [otid for otid in other_ids[dup_mask].tolist() if otid > cur_id]
                    )
                    dup_ids = [str(otid) for otid in dup_ids]
                    if len(dup_ids) > 0:
                        dup_str = ",".join(dup_ids)
                        dup_str = str(cur_id) + " : [" + dup_str + "],"
                        print(dup_str)

    # add ego at index 0
    ego_poses = ego_poses[np.newaxis]
    traj_poses = np.concatenate([ego_poses, traj_poses], axis=0)
    traj_valid = np.concatenate(
        [np.ones((1, ego_poses.shape[1]), dtype=bool), traj_valid], axis=0
    )
    traj_lwh = np.concatenate([np.array([NV_EGO_LWH]), traj_lwh], axis=0)
    traj_ids = np.array(["ego"] + traj_ids.tolist())

    if map_tile is not None:
        traj_poses[:, :, :2, -1] += map_tile.trans_offset

    return traj_poses, traj_valid, traj_lwh, ego_pose_timestamps * 1e-6, traj_ids


if __name__ == "__main__":
    tile_path = "./data/nvidia/nvmaps/92d651e5-21d2-4816-b16d-0feace622aa1/jsv3/92d651e5-21d2-4816-b16d-0feace622aa1/tile/4bd02829-cab6-435d-8ebd-679c96787f8b_json"
    tile = load_tile(tile_path, layers=["lane_dividers_v1", "lane_channels_v1"])
    # convert to nuscenes-like map format if desired
    nusc_tile = convert_tile_to_nuscenes(tile)

    # dynamic_obj_path = './data/nvidia/dynamic_object_poses.npz'
    # dyn_obj_data = np.load(dynamic_obj_path)
    # dyn_obj_poses = dyn_obj_data['poses']
    # dyn_obj_timestamps = dyn_obj_data['pose_timestamps']

    # print(dyn_obj_poses.shape)

    # # debug_viz_vid(tile.segments, 'dyn_obj',
    # #                 comp_out_path='./out/dev_gtc_demo/dev_preprocess',
    # #                 poses=dyn_obj_poses[np.newaxis],
    # #                 poses_valid=np.ones((1, dyn_obj_poses.shape[0])),
    # #                 poses_lwh=np.array([[5.087, 2.307, 1.856]]),
    # #                 # pose_ids=traj_ids,
    # #                 subsamp=1,
    # #                 fps=30)

    # # TODO option to just load specific ids?

    # # rivermark
    # autolabels_path = './data/nvidia/endeavor/labels/autolabels.pkl'
    # ego_images_path = './data/nvidia/ego_session/processed/44093/images/image_00'
    # lidar_rig_path = './data/nvidia/endeavor/poses/T_lidar_rig.npz'
    # frame_range = (510, 880)
    # traj_poses, traj_valid, traj_lwh, traj_t, traj_ids = load_trajectories(autolabels_path, ego_images_path, lidar_rig_path,
    #                                                                         frame_range=frame_range,
    #                                                                         postprocess=True,
    #                                                                         # extra_obj_path=dynamic_obj_path,
    #                                                                         # load_ids=['ego', 'extra'],
    #                                                                         crop2valid=False,
    #                                                                         fill_first_n=160)

    # print(traj_poses.shape)
    # debug_viz_vid(tile.segments, 'rivermark_fill',
    #                 comp_out_path='./out/dev_gtc_demo/dev_preprocess',
    #                 poses=traj_poses,
    #                 poses_valid=traj_valid,
    #                 poses_lwh=traj_lwh,
    #                 pose_ids=traj_ids,
    #                 subsamp=1,
    #                 fps=30)

    # exit()

    autolabels_path = "./data/nvidia/endeavor/labels/autolabels.pkl"
    ego_images_path = "./data/nvidia/endeavor/images/image_00"
    lidar_rig_path = "./data/nvidia/endeavor/poses/T_lidar_rig.npz"

    # this just filters which frames to process, can be None
    # frame_range = (3170, 3590) # (3200, 3545) # (0, 900), (1000, 2400), None
    # frame_range = (60, 3590)
    frame_range = (3000, 3590)
    # frame_range = (2000, 2400)
    # frame_range = (570, 870) # merge
    # frame_range = (240, 660) # merge
    # frame_range = (1440, 1860) # merge

    # process on the original nvmap
    traj_poses, traj_valid, traj_lwh, traj_t, traj_ids = load_trajectories(
        autolabels_path,
        ego_images_path,
        lidar_rig_path,
        frame_range=frame_range,
        postprocess=True,
        mine_dups=False,
        fill_first_n=300,
    )
    print(traj_poses.shape)
    debug_viz_vid(
        tile.segments,
        "extrap_proc_ids_sframe_00003000_eframe_00003300",
        comp_out_path="./out/dev_gtc_demo/dev_preprocess",
        poses=traj_poses,
        poses_valid=traj_valid,
        poses_lwh=traj_lwh,
        pose_ids=traj_ids,
        subsamp=1,
        fps=30,
    )

    # # process with the nuscenes map version
    # # (the only difference is the coordinate system is offset such that the origin is at bottom left)
    # traj_poses, traj_valid, traj_lwh, traj_t, traj_ids = load_trajectories(autolabels_path, ego_images_path, lidar_rig_path,
    #                                                                     frame_range=frame_range,
    #                                                                     map_tile=nusc_tile,
    #                                                                     postprocess=True)
    # debug_viz_vid(nusc_tile.segments, 'processed_sframe_00003200_eframe_00003545_nusc',
    #                 comp_out_path='./out/gtc_demo/dev_preprocess',
    #                 poses=traj_poses,
    #                 poses_valid=traj_valid,
    #                 poses_lwh=traj_lwh,
    #                 pose_ids=traj_ids,
    #                 subsamp=3,
    #                 fps=10)

    #
    # output single pose for Amlan
    #

    # single_frame_idx = 3500
    # last_pose = traj_poses[:,single_frame_idx:single_frame_idx+1]
    # last_valid = traj_valid[:,single_frame_idx]
    # last_step_valid_poses = last_pose[last_valid]
    # print(last_step_valid_poses.shape)
    # last_step_lwh = traj_lwh[last_valid]

    # debug_viz_segs(nusc_tile.segments, 'endeavor_nusc_step%d' % (single_frame_idx), poses=last_step_valid_poses, poses_valid=traj_valid[:,single_frame_idx:single_frame_idx+1][last_valid], poses_lwh=last_step_lwh)

    # np.savez('./out/dev_nvmap/endeavor_poses_frame%06d.npz' % (single_frame_idx),
    #             poses=last_step_valid_poses,
    #             lwh=last_step_lwh)
    # exit()

    #
    # Output GPS trajectories for Jeremy
    #

    # #  convert ego poses back to global ECEF coordinate system
    # N, T, _, _ = traj_poses.shape
    # ecef_traj_poses = np.matmul(GLOBAL_BASE_POSE[np.newaxis,np.newaxis], traj_poses)
    # print(ecef_traj_poses.shape)
    # # convert to GPS
    # gps_traj_poses = ecef_2_lat_lng_alt(ecef_traj_poses.reshape((N*T, 4, 4)), earth_model='WGS84')
    # lat_lng_alt, orientation_axis, orientation_angle = gps_traj_poses

    # save_path = os.path.join('./out/dev_nvmap/endeavor_trajectory_track_ids.npz')
    # out_dict = {
    #     'timestamps' : traj_t,
    #     'track_ids' : traj_ids,
    #     'ecef_poses' : ecef_traj_poses,
    #     'pose_valid' : traj_valid,
    #     'gps_lat_lng_alt'  : lat_lng_alt.reshape((N, T, 3)),
    #     'gps_orientation_axis' : orientation_axis.reshape((N, T, 3)),
    #     'gps_orientation_angle_degrees' : orientation_angle.reshape((N, T, 1))
    # }
    # for k, v in out_dict.items():
    #     print(k)
    #     print(v.shape)
    #     # if k != 'timestamps':
    #     #     print(np.sum(np.isnan(v), axis=1))
    # np.savez(save_path, **out_dict)

    # exit()

    # debug_viz_vid(tile.segments, 'gt_sframe_00003200_eframe_00003545',
    #                 comp_out_path='./out/gtc_demo/dev_preprocess',
    #                 poses=traj_poses,
    #                 poses_valid=traj_valid,
    #                 poses_lwh=traj_lwh,
    #                 subsamp=3,
    #                 fps=10)
