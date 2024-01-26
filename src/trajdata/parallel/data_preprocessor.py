import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from torch.utils.data import Dataset

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import Scene, SceneMetadata
from trajdata.utils import agent_utils, env_utils


def scene_paths_collate_fn(filled_scenes: List) -> List:
    return filled_scenes


class ParallelDatasetPreprocessor(Dataset):
    def __init__(
        self,
        scene_info_list: List[SceneMetadata],
        envs_dir_dict: Dict[str, str],
        env_cache_path: str,
        desired_dt: Optional[float],
        cache_class: Type[SceneCache],
        rebuild_cache: bool,
        env_dict
    ) -> None:
        self.env_cache_path = np.array(env_cache_path).astype(np.string_)
        self.desired_dt = desired_dt
        self.cache_class = cache_class
        self.rebuild_cache = rebuild_cache

        env_names: List[str] = list(envs_dir_dict.keys())
        scene_idxs_names: List[Tuple[int, str]] = [
            (idx, scene_info.name) for idx, scene_info in enumerate(scene_info_list)
        ]
        scene_name_idxs, scene_names = zip(*scene_idxs_names)

        self.scene_idxs = np.array(
            [scene_info.raw_data_idx for scene_info in scene_info_list], dtype=int
        )
        self.env_name_idxs = np.array(
            [env_names.index(scene_info.env_name) for scene_info in scene_info_list],
            dtype=int,
        )

        self.scene_name_idxs = np.array(scene_name_idxs, dtype=int)
        self.env_names_arr = np.array(env_names).astype(np.string_)
        self.scene_names_arr = np.array(scene_names).astype(np.string_)
        self.data_dir_arr = np.array(list(envs_dir_dict.values())).astype(np.string_)

        self.data_len: int = len(scene_info_list)
        self.env_dict = env_dict

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, idx: int) -> str:

        env_cache_path: Path = Path(str(self.env_cache_path, encoding="utf-8"))
        env_cache: EnvCache = EnvCache(env_cache_path)

        env_idx: int = self.env_name_idxs[idx]
        scene_idx: int = self.scene_name_idxs[idx]

        scene_start_time = time.time()
        env_name: str = str(self.env_names_arr[env_idx], encoding="utf-8")

        if 'nuplan' not in env_name:
            raw_dataset = env_utils.get_raw_dataset(
                env_name, str(self.data_dir_arr[env_idx], encoding="utf-8")
            )
        else:
            raw_dataset = self.env_dict[env_name]
            
        scene_name: str = str(self.scene_names_arr[scene_idx], encoding="utf-8")

        scene_prepare_time = time.time()
        print(f"Process {os.getpid()}: {scene_name} prepare took {scene_prepare_time - scene_start_time} seconds.")


        scene_info = SceneMetadata(
            env_name, scene_name, raw_dataset.metadata.dt, self.scene_idxs[idx]
        )


        # Leaving verbose False here so that we don't spam
        # stdout with loading messages.
        # if 'nuplan' in env_name:
        #     raw_dataset.load_dataset_obj(verbose=True, scenes=self.env_scences[env_name])
        if 'nuplan' not in env_name:
            raw_dataset.load_dataset_obj(verbose=False)

        try:
            scene: Scene = agent_utils.get_agent_data(
                scene_info,
                raw_dataset,
                env_cache,
                self.rebuild_cache,
                self.cache_class,
                self.desired_dt,
            )
        except ValueError as e:
            print(f"Process {os.getpid()}: {scene_name} failed with error: {e}")
            return None

        if 'nuplan' not in env_name:
            raw_dataset.del_dataset_obj()

        scene_end_time = time.time()
        print(f"Process {os.getpid()}: {scene_name} finish all took {scene_end_time - scene_start_time} seconds.")

        if scene is None:
            # This provides an escape hatch in case there's a reason we
            # don't want to add a scene to the list of scenes. As an example,
            # nuPlan has a scene with only a single frame of data which we
            # can't do much with in terms of prediction/planning/etc.
            return None

        scene_path: Path = EnvCache.scene_metadata_path(
            env_cache.path, scene.env_name, scene.name, scene.dt
        )
        return str(scene_path)
