import numpy as np

ALLOWED_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Simplified BEV IoU for quick sanity-check evaluation.
    box = [x, y, z, dx, dy, dz, yaw]
    Yaw is ignored and the IoU is computed axis-aligned in x/y.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ious = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    for i, a in enumerate(boxes_a):
        ax1 = a[0] - a[3] / 2.0
        ay1 = a[1] - a[4] / 2.0
        ax2 = a[0] + a[3] / 2.0
        ay2 = a[1] + a[4] / 2.0

        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)

        for j, b in enumerate(boxes_b):
            bx1 = b[0] - b[3] / 2.0
            by1 = b[1] - b[4] / 2.0
            bx2 = b[0] + b[3] / 2.0
            by2 = b[1] + b[4] / 2.0

            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)

            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter = inter_w * inter_h

            union = a_area + b_area - inter
            if union > 0:
                ious[i, j] = inter / union

    return ious


def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)


def eval_class(gt_annos, det_annos, class_name, iou_thresh=0.5):
    gt_per_sample = []
    total_gt = 0

    for gt in gt_annos:
        mask = np.array(gt["name"]) == class_name
        boxes = gt["gt_boxes_lidar"][mask]
        gt_per_sample.append({
            "boxes": boxes,
            "matched": np.zeros(len(boxes), dtype=bool),
        })
        total_gt += len(boxes)

    all_dets = []
    for sample_idx, det in enumerate(det_annos):
        names = np.array(det.get("name", []))
        boxes = np.array(det.get("boxes_lidar", np.zeros((0, 7), dtype=np.float32)))
        scores = np.array(det.get("score", np.zeros((0,), dtype=np.float32)))

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

    all_dets.sort(key=lambda x: x[1], reverse=True)

    tp = np.zeros(len(all_dets), dtype=np.float32)
    fp = np.zeros(len(all_dets), dtype=np.float32)

    for i, (sample_idx, score, pred_box) in enumerate(all_dets):
        gt_info = gt_per_sample[sample_idx]
        gt_boxes = gt_info["boxes"]
        matched = gt_info["matched"]

        if len(gt_boxes) == 0:
            fp[i] = 1.0
            continue

        ious = boxes_iou_bev(np.expand_dims(pred_box, axis=0), gt_boxes)[0]
        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_thresh and not matched[best_gt_idx]:
            tp[i] = 1.0
            gt_info["matched"][best_gt_idx] = True
        else:
            fp[i] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    if total_gt > 0:
        recalls = cum_tp / total_gt
    else:
        recalls = np.zeros_like(cum_tp)

    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
    ap = compute_ap(recalls, precisions)

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
    used_classes = [c for c in class_names if c in ALLOWED_CLASSES]

    results = {}
    lines = []
    aps = []

    lines.append("CarlaNuScenes6 evaluation (BEV axis-aligned IoU)")
    lines.append("")

    for cls in used_classes:
        cls_result = eval_class(gt_annos, det_annos, cls, iou_thresh=0.5)
        results[f"{cls}_ap"] = cls_result["ap"]
        results[f"{cls}_precision"] = cls_result["precision"]
        results[f"{cls}_recall"] = cls_result["recall"]
        results[f"{cls}_num_gt"] = cls_result["num_gt"]
        results[f"{cls}_num_det"] = cls_result["num_det"]

        aps.append(cls_result["ap"])

        lines.append(
            f"{cls:12s} | AP={cls_result['ap']:.4f} | "
            f"P={cls_result['precision']:.4f} | "
            f"R={cls_result['recall']:.4f} | "
            f"GT={cls_result['num_gt']} | DET={cls_result['num_det']}"
        )

    mean_ap = float(np.mean(aps)) if len(aps) > 0 else 0.0
    results["mAP"] = mean_ap

    lines.append("")
    lines.append(f"mAP={mean_ap:.4f}")

    return "\n".join(lines), results
