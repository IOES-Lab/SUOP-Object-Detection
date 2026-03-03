#png_make.py
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Rotation angle settings (degrees)
ROT_X_DEG = -5
ROT_Y_DEG = -170
ROT_Z_DEG = 180

def rotation_matrix(axis: str, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0,  0],
                         [0, c, -s],
                         [0, s,  c]])
    if axis == 'y':
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])
    if axis == 'z':
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]])
    raise ValueError(f"Unknown axis '{axis}'")

def save_single_view(points: np.ndarray, output_path: str):
    Rx = rotation_matrix('x', np.deg2rad(ROT_X_DEG))
    Ry = rotation_matrix('y', np.deg2rad(ROT_Y_DEG))
    Rz = rotation_matrix('z', np.deg2rad(ROT_Z_DEG))
    R  = Rz @ Ry @ Rx

    pts = points @ R.T

    z = pts[:, 2]
    zmin, zmax = z.min(), z.max()
    norm = (z - zmin) / (zmax - zmin + 1e-8)
    clipped = np.clip((norm - 0.3) / 0.8, 0.0, 1.0)
    cmap  = plt.get_cmap('jet')
    cols  = cmap(clipped)[:, :3]

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=cols, s=0.1, depthshade=False)
    ax.view_init(elev=30, azim=45)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_distance_token(path_parts):
    for p in path_parts:
        pl = p.lower()
        if "10m" in pl:
            return "10m"
        if "6m" in pl:
            return "6m"
        if "3m" in pl:
            return "3m"
    return None

if __name__ == "__main__":
# TODO: Update these paths for your environment
    src_root  = "/path/to/..."
    save_root = "/path/to/..."

    out_3m  = os.path.join(save_root, "3m")
    out_6m  = os.path.join(save_root, "6m")
    out_10m = os.path.join(save_root, "10m")
    os.makedirs(out_3m, exist_ok=True)
    os.makedirs(out_6m, exist_ok=True)
    os.makedirs(out_10m, exist_ok=True)

    for root, _, files in os.walk(src_root):
        rel_parts = os.path.relpath(root, src_root).split(os.sep)

        dist = get_distance_token(rel_parts)
        if dist not in {"3m", "6m", "10m"}:
            continue

        for fname in files:
            if not fname.lower().endswith(".ply"):
                continue

            ply_path = os.path.join(root, fname)
            pcd = o3d.io.read_point_cloud(ply_path)
            pts = np.asarray(pcd.points)
            if pts.size == 0:
                continue

            stem = os.path.splitext(fname)[0]
            out_name = f"{stem}.png"

            if dist == "3m":
                out_dir = out_3m
            elif dist == "6m":
                out_dir = out_6m
            else:
                out_dir = out_10m

            out_png = os.path.join(out_dir, out_name)

            if os.path.exists(out_png):
                print(f"[SKIP] exists: {out_png}")
                continue

            save_single_view(pts, out_png)
            print(f"[OK] Saved {out_png}")

    print("[DONE] 3m/6m/10m PLY -> PNG")