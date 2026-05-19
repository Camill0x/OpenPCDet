import numpy as np

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

ALLOWED_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]
DEFAULT_IOU_THRESHOLDS = {
    "car": 0.5,
    "truck": 0.5,
    "bus": 0.5,
    "motorcycle": 0.5,
    "bicycle": 0.5,
    "pedestrian": 0.5,
}


def boxes_iou_3d(boxes_a, boxes_b):
    """
    Full 3D IoU for rotated boxes.
    box = [x, y, z, dx, dy, dz, yaw]
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)[:, :7]
    boxes_b = np.asarray(boxes_b, dtype=np.float32)[:, :7]

    bev_iou = boxes_bev_iou_cpu(boxes_a, boxes_b)

    area_a = (boxes_a[:, 3] * boxes_a[:, 4])[:, None]
    area_b = (boxes_b[:, 3] * boxes_b[:, 4])[None, :]
    bev_intersection = bev_iou * (area_a + area_b) / np.maximum(1.0 + bev_iou, 1e-6)

    a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2.0)[:, None]
    a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2.0)[:, None]
    b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2.0)[None, :]
    b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2.0)[None, :]
    height_overlap = np.clip(
        np.minimum(a_height_max, b_height_max) - np.maximum(a_height_min, b_height_min),
        a_min=0.0,
        a_max=None,
    )

    intersection_3d = bev_intersection * height_overlap
    volume_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5])[:, None]
    volume_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5])[None, :]
    union_3d = np.maximum(volume_a + volume_b - intersection_3d, 1e-6)

    return intersection_3d / union_3d


def compute_interpolated_ap(recalls, precisions, num_sample_points=101):
    """
    Compute interpolated AP on a fixed recall grid.
    This is more stable than reporting the raw final precision.
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    recall_grid = np.linspace(0.0, 1.0, num_sample_points)
    interpolated = np.zeros_like(recall_grid, dtype=np.float32)

    for idx, recall_threshold in enumerate(recall_grid):
        mask = recalls >= recall_threshold
        interpolated[idx] = np.max(precisions[mask]) if np.any(mask) else 0.0

    return float(np.mean(interpolated))


def eval_class(gt_annos, det_annos, class_name, iou_thresh):
    gt_per_sample = []
    total_gt = 0

    for gt in gt_annos:
        mask = np.array(gt["name"]) == class_name
        boxes = np.asarray(gt["gt_boxes_lidar"][mask], dtype=np.float32)
        gt_per_sample.append(
            {
                "boxes": boxes,
                "matched": np.zeros(len(boxes), dtype=bool),
            }
        )
        total_gt += len(boxes)

    all_dets = []
    for sample_idx, det in enumerate(det_annos):
        names = np.array(det.get("name", []))
        boxes = np.asarray(det.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32)), dtype=np.float32)
        scores = np.asarray(det.get("score", np.zeros((0,), dtype=np.float32)), dtype=np.float32)

        mask = names == class_name
        for box, score in zip(boxes[mask], scores[mask]):
            all_dets.append((sample_idx, float(score), box))

    if len(all_dets) == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_gt": int(total_gt),
            "num_det": 0,
        }

    all_dets.sort(key=lambda item: item[1], reverse=True)

    tp = np.zeros(len(all_dets), dtype=np.float32)
    fp = np.zeros(len(all_dets), dtype=np.float32)

    # Cache IoU matrices per sample to avoid recomputing for every detection.
    det_boxes_per_sample = {}
    det_indices_per_sample = {}
    for det_idx, (sample_idx, _score, box) in enumerate(all_dets):
        det_boxes_per_sample.setdefault(sample_idx, []).append(box)
        det_indices_per_sample.setdefault(sample_idx, []).append(det_idx)

    iou_cache = {}
    for sample_idx, sample_boxes in det_boxes_per_sample.items():
        gt_boxes = gt_per_sample[sample_idx]["boxes"]
        sample_boxes = np.asarray(sample_boxes, dtype=np.float32)
        iou_cache[sample_idx] = boxes_iou_3d(sample_boxes, gt_boxes)

    sample_offsets = {sample_idx: 0 for sample_idx in det_boxes_per_sample}

    for sample_idx, _score, _pred_box in all_dets:
        local_det_idx = sample_offsets[sample_idx]
        sample_offsets[sample_idx] += 1

        det_global_idx = det_indices_per_sample[sample_idx][local_det_idx]
        gt_info = gt_per_sample[sample_idx]
        gt_boxes = gt_info["boxes"]
        matched = gt_info["matched"]

        if len(gt_boxes) == 0:
            fp[det_global_idx] = 1.0
            continue

        ious = iou_cache[sample_idx][local_det_idx]
        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_thresh and not matched[best_gt_idx]:
            tp[det_global_idx] = 1.0
            gt_info["matched"][best_gt_idx] = True
        else:
            fp[det_global_idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    if total_gt > 0:
        recalls = cum_tp / total_gt
    else:
        recalls = np.zeros_like(cum_tp)

    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
    ap = compute_interpolated_ap(recalls, precisions)

    final_precision = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0

    return {
        "ap": ap,
        "precision": final_precision,
        "recall": final_recall,
        "num_gt": int(total_gt),
        "num_det": int(len(all_dets)),
    }


def carla_nuscenes6_eval_result(gt_annos, det_annos, class_names, eval_metric="kitti"):
    used_classes = [class_name for class_name in class_names if class_name in ALLOWED_CLASSES]

    results = {}
    lines = []
    precisions = []
    recalls = []
    aps = []

    lines.append("CarlaNuScenes6 evaluation (3D IoU)")
    lines.append("")

    for class_name in used_classes:
        class_result = eval_class(
            gt_annos=gt_annos,
            det_annos=det_annos,
            class_name=class_name,
            iou_thresh=DEFAULT_IOU_THRESHOLDS[class_name],
        )
        results[f"{class_name}_ap"] = class_result["ap"]
        results[f"{class_name}_precision"] = class_result["precision"]
        results[f"{class_name}_recall"] = class_result["recall"]
        results[f"{class_name}_num_gt"] = class_result["num_gt"]
        results[f"{class_name}_num_det"] = class_result["num_det"]

        precisions.append(class_result["precision"])
        recalls.append(class_result["recall"])
        aps.append(class_result["ap"])

        lines.append(
            f"{class_name:12s} | AP={class_result['ap']:.4f} | "
            f"P={class_result['precision']:.4f} | "
            f"R={class_result['recall']:.4f} | "
            f"GT={class_result['num_gt']} | DET={class_result['num_det']}"
        )

    macro_precision = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    macro_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
    mean_ap = float(np.mean(aps)) if len(aps) > 0 else 0.0

    results["macro_precision"] = macro_precision
    results["macro_recall"] = macro_recall
    results["mAP"] = mean_ap

    lines.append("")
    lines.append(f"macro_precision={macro_precision:.4f}")
    lines.append(f"macro_recall={macro_recall:.4f}")
    lines.append(f"mAP={mean_ap:.4f}")

    return "\n".join(lines), results
