import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

def random_jitter(points, sigma=0.01, clip=0.02):
    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + jitter

def random_rotation(points, max_angle=np.pi/12):
    theta = np.random.uniform(-max_angle, max_angle)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    return points @ rot

def random_sample(points, num_points):
    N = points.shape[0]
    if N == num_points:
        return points
    elif N > num_points:
        idx = np.random.choice(N, num_points, replace=False)
        return points[idx]
    else:
        idx = np.random.choice(N, num_points, replace=True)
        return points[idx]

class DetectionDataset(Dataset):
    def __init__(self, h5_path, num_points=2048, train_mode=True, class_names=None):
        with h5py.File(h5_path, 'r') as f:
            pts = f['points'][:]
            bbs = f['bboxes'][:]
            mask = f['bbox_mask'][:]
            cls = f.get('classes', None)
            if cls is not None:
                cls = f['classes'][:]
        
        if pts.ndim == 3 and pts.shape[2] > 3:
            pts = pts[..., :3]
        
        self.points = pts.astype(np.float32)
        self.bboxes = bbs.astype(np.float32)
        self.mask = mask.astype(np.int32)
        
        if cls is not None:
            self.classes = cls.astype(np.int64)
        else:
            if class_names is None:
                self.classes = np.zeros(len(self.points), dtype=np.int64)
            else:
                base_name = os.path.basename(h5_path).lower()
                for i, cname in enumerate(class_names):
                    if cname in base_name:
                        class_idx = i
                        break
                else:
                    class_idx = 0
                self.classes = np.full(len(self.points), class_idx, dtype=np.int64)
        
        self.num_points = num_points
        self.train_mode = train_mode

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pts = self.points[idx].astype(np.float32)
        bbs = self.bboxes[idx].astype(np.float32)
        m = self.mask[idx]

        pts = random_sample(pts, self.num_points)

        if self.train_mode:
            pts = random_jitter(pts, sigma=0.01, clip=0.02)
            pts = random_rotation(pts, max_angle=np.pi/12)

        valid = np.where(m == 1)[0]
        if len(valid) > 0:
            bbox = bbs[valid[0]].astype(np.float32)
        else:
            bbox = np.zeros(6, dtype=np.float32)

        out = [torch.from_numpy(pts).float(), torch.from_numpy(bbox).float()]
        if self.classes is not None:
            out.append(torch.tensor(self.classes[idx], dtype=torch.long))
        return tuple(out)
