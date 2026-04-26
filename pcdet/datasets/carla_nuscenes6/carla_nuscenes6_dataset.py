import copy
import pickle
from pathlib import Path

import numpy as np
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


class CarlaNuScenes6Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = Path(root_path if root_path is not None else dataset_cfg.DATA_PATH)
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / "ImageSets" / f"{self.split}.txt"
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []

        self.infos = []
        self.include_carla_nuscenes6_data(self.mode)

    def include_carla_nuscenes6_data(self, mode):
        if self.logger is not None:
            self.logger.info("Loading CarlaNuScenes6 dataset")

        dataset_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue

            with open(info_path, "rb") as f:
                infos = pickle.load(f)
                dataset_infos.extend(infos)

        self.infos.extend(dataset_infos)

        if self.logger is not None:
            self.logger.info(f"Total samples for CarlaNuScenes6 dataset: {len(dataset_infos)}")

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger,
        )
        self.split = split
        split_dir = self.root_path / "ImageSets" / f"{self.split}.txt"
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []

    def get_lidar(self, sample_idx):
        info = self.infos[sample_idx] if isinstance(sample_idx, int) else None
        if info is None:
            raise ValueError("get_lidar expects dataset index, not frame id")

        lidar_path = self.root_path / info["point_cloud"]["lidar_path"]
        points = np.load(lidar_path).astype(np.float32)

        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(f"Invalid point cloud shape {points.shape} at {lidar_path}")

        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        points = np.load(self.root_path / info["point_cloud"]["lidar_path"]).astype(np.float32)

        if self.dataset_cfg.get('APPEND_ZERO_TIMESTAMP', False):
            timestamps = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack([points, timestamps])

        input_dict = {
            "frame_id": info["frame_id"],
            "points": points,
        }

        if "annos" in info:
            annos = info["annos"]
            gt_names = annos["name"]
            gt_boxes_lidar = annos["gt_boxes_lidar"].astype(np.float32)

            if self.dataset_cfg.get("APPEND_ZERO_VELOCITY_TO_GT", False):
                if gt_boxes_lidar.shape[1] == 7:
                    zeros_vel = np.zeros((gt_boxes_lidar.shape[0], 2), dtype=np.float32)
                    gt_boxes_lidar = np.hstack([gt_boxes_lidar, zeros_vel])

            input_dict.update(
                {
                    "gt_names": gt_names,
                    "gt_boxes": gt_boxes_lidar,
                }
            )

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        def get_template_prediction(num_samples):
            ret_dict = {
                "name": np.zeros(num_samples, dtype=object),
                "score": np.zeros(num_samples),
                "boxes_lidar": np.zeros((num_samples, 7)),
                "pred_labels": np.zeros(num_samples, dtype=np.int64),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()

            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict["name"] = np.array(class_names)[pred_labels - 1]
            pred_dict["score"] = pred_scores
            pred_dict["boxes_lidar"] = pred_boxes
            pred_dict["pred_labels"] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict["frame_id"][index]
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict["frame_id"] = frame_id
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if len(self.infos) == 0 or "annos" not in self.infos[0]:
            return "No ground-truth boxes for evaluation", {}

        try:
            from .carla_nuscenes6_eval import carla_nuscenes6_eval_result
        except ImportError:
            return "carla_nuscenes6_eval.py not found yet", {}

        eval_gt_annos = [copy.deepcopy(info["annos"]) for info in self.infos]
        ap_result_str, ap_dict = carla_nuscenes6_eval_result(
            gt_annos=eval_gt_annos,
            det_annos=det_annos,
            class_names=class_names,
            eval_metric=self.dataset_cfg.get("EVAL_METRIC", "kitti"),
        )
        return ap_result_str, ap_dict

    def create_groundtruth_database(self, info_path=None, used_classes=None, split="train"):
        import torch

        database_save_path = self.root_path / f"gt_database_{split}"
        db_info_save_path = self.root_path / f"carla_nuscenes6_dbinfos_{split}.pkl"

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, "rb") as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            info = infos[k]
            sample_idx = info["frame_id"]
            print(f"gt_database sample: {k + 1}/{len(infos)}")

            points = np.load(self.root_path / info["point_cloud"]["lidar_path"]).astype(np.float32)
            annos = info["annos"]
            names = annos["name"]
            gt_boxes = annos["gt_boxes_lidar"]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]),
                torch.from_numpy(gt_boxes[:, 0:7]),
            ).numpy()

            for i in range(num_obj):
                filename = f"{sample_idx}_{names[i]}_{i}.bin"
                filepath = database_save_path / filename

                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]

                with open(filepath, "wb") as f:
                    gt_points.tofile(f)

                if used_classes is None or names[i] in used_classes:
                    db_info = {
                        "name": names[i],
                        "path": str(filepath.relative_to(self.root_path)),
                        "image_idx": sample_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                    }
                    all_db_infos.setdefault(names[i], []).append(db_info)

        for k, v in all_db_infos.items():
            print(f"Database {k}: {len(v)}")

        with open(db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)
