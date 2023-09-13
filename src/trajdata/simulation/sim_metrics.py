from typing import Dict

import numpy as np
import pandas as pd

from shapely import affinity
from shapely.geometry import Polygon

class SimMetric:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        raise NotImplementedError()


class ADE(SimMetric):
    def __init__(self) -> None:
        super().__init__("ade")

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        err_df = pd.DataFrame(index=gt_df.index, columns=["error"])
        err_df["error"] = np.linalg.norm(gt_df[["x", "y"]] - sim_df[["x", "y"]], axis=1)
        return err_df.groupby("agent_id")["error"].mean().to_dict()


class FDE(SimMetric):
    def __init__(self) -> None:
        super().__init__("fde")

    def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, float]:
        err_df = pd.DataFrame(index=gt_df.index, columns=["error"])
        err_df["error"] = np.linalg.norm(gt_df[["x", "y"]] - sim_df[["x", "y"]], axis=1)
        return err_df.groupby("agent_id")["error"].last().to_dict()

class CrashDetect(SimMetric):
  def __init__(self, tgt_agent_ids, agent_extends, iou_threshold=0.1, mode='sim') -> None:
    super().__init__("crach_detect")
    
    self.tgt_agent_ids = tgt_agent_ids
    self.agent_extends = agent_extends
    self.iou_threshold = iou_threshold
    self.mode = mode

  def _get_box_polygon(self, agent, extents):
    box_points = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    box_points[:, 0] = box_points[:, 0] * extents[1]
    box_points[:, 1] = box_points[:, 1] * extents[0]
    box = Polygon(box_points)

    # Get the agent polygon
    box = affinity.rotate(box, agent['heading'], origin='centroid')
    box = affinity.translate(box, agent['x'], agent['y'])
    return box

  def _poly_iou_check(self, poly1, poly2, iou_threshold):
    if not poly1.intersects(poly2):
      return False

    union = poly1.union(poly2).area
    inter = poly1.intersection(poly2).area
    iou = inter / union

    return iou > iou_threshold

  def __call__(self, gt_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict[str, bool]:
    data = sim_df if self.mode == 'sim' else gt_df

    all_scene_ts = data.index.get_level_values('scene_ts').unique()

    crash_logs = {agent_id: [] for agent_id in self.tgt_agent_ids}

    for scene_ts in all_scene_ts:
      scene_data = data.xs(scene_ts, level='scene_ts')
      frame_agent_ids = scene_data.index.get_level_values('agent_id').unique()

      # create polygons for all agents in this frame
      agent_boxes = {}
      for agent_id in frame_agent_ids:
        agent = scene_data.loc[agent_id]

        # Get the extents for the agent
        agent_extents = self.agent_extends[agent_id]

        # Get the box polygon
        box = self._get_box_polygon(agent, agent_extents)
        agent_boxes[agent_id] = box

      for target_id in self.tgt_agent_ids:
        # target agent does not exist in this frame
        if target_id not in frame_agent_ids:
          crash_logs[target_id].append(0)
          continue

        target_box = agent_boxes[target_id]

        for agent_id in frame_agent_ids:
          if agent_id == target_id:
            continue

          agent_box = agent_boxes[agent_id]

          if self._poly_iou_check(target_box, agent_box, self.iou_threshold):
            crash_logs[target_id].append(1)
            break

        if scene_ts not in crash_logs[target_id]:
          crash_logs[target_id].append(0)
    
    crash_detect = {}
    for agent_id in self.tgt_agent_ids:
      crash_detect[agent_id] = crash_logs[agent_id].count(1) > 0

    return crash_detect