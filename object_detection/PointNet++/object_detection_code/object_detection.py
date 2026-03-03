import torch
from model import PointNet2Detection
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================
# IoU calculation (AABB, center/scale format)
# ======================
def iou_3d(box1, box2):
    box1_min = box1[:3] - box1[3:] / 2
    box1_max = box1[:3] + box1[3:] / 2
    box2_min = box2[:3] - box2[3:] / 2
    box2_max = box2[:3] + box2[3:] / 2

    inter_min = np.maximum(box1_min, box2_min)
    inter_max = np.minimum(box1_max, box2_max)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)

    vol1 = np.prod(box1[3:])
    vol2 = np.prod(box2[3:])
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

# ======================
# Visualization functions
# ======================
def draw_scene(points, gt_bbox, pred_bbox, iou, scene_id):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Scene {scene_id+1} | IoU: {iou:.4f}")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=2, alpha=0.3)

    def draw_bbox(bbox, color):
        cx, cy, cz, dx, dy, dz = bbox
        corners = np.array([
            [cx - dx/2, cy - dy/2, cz - dz/2],
            [cx + dx/2, cy - dy/2, cz - dz/2],
            [cx + dx/2, cy + dy/2, cz - dz/2],
            [cx - dx/2, cy + dy/2, cz - dz/2],
            [cx - dx/2, cy - dy/2, cz + dz/2],
            [cx + dx/2, cy - dy/2, cz + dz/2],
            [cx + dx/2, cy + dy/2, cz + dz/2],
            [cx - dx/2, cy + dy/2, cz + dz/2],
        ])
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        for e in edges:
            ax.plot(*zip(corners[e[0]], corners[e[1]]), color=color)

    draw_bbox(gt_bbox, 'green')
    draw_bbox(pred_bbox, 'red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# ======================
# Model loading
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PointNet2Detection(num_classes=5).to(device)
model.load_state_dict(torch.load("all_object_detector.pth", map_location=device))
model.eval()

# ======================
# Data loading
# ======================
with h5py.File("tire_val.h5", "r") as f:
    points = f["points"][:]      # (N, 2048, 3)
    labels = f["classes"][:]     # (N,)
    bboxes = f["bboxes"][:]      # (N, 6)

# ======================
# Prediction and Analysis
# ======================
with torch.no_grad():
    for i in range(len(points)):
        pt = torch.tensor(points[i]).unsqueeze(0).float().to(device)
        pred_bbox_tensor, pred_logits = model(pt)
        pred_cls = pred_logits.argmax(dim=1).item()
        pred_bbox_np = pred_bbox_tensor.squeeze(0).cpu().numpy()

        gt_cls = labels[i]
        gt_bbox_raw = bboxes[i][0]  
        cx = (gt_bbox_raw[0] + gt_bbox_raw[3]) / 2
        cy = (gt_bbox_raw[1] + gt_bbox_raw[4]) / 2
        cz = (gt_bbox_raw[2] + gt_bbox_raw[5]) / 2
        dx = gt_bbox_raw[3] - gt_bbox_raw[0]
        dy = gt_bbox_raw[4] - gt_bbox_raw[1]
        dz = gt_bbox_raw[5] - gt_bbox_raw[2]
        gt_bbox = np.array([cx, cy, cz, dx, dy, dz])

        iou = iou_3d(gt_bbox, pred_bbox_np)

        print(f"[Scene {i+1:03d}] GT class: {gt_cls} | Pred class: {pred_cls} | IoU: {iou:.4f}")
        print(f"  └ GT BBox  : {gt_bbox}")
        print(f"  └ Pred BBox: {pred_bbox_np}")
        print("-" * 80)

        draw_scene(points[i], gt_bbox, pred_bbox_np, iou, i)
