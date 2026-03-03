import os
import numpy as np
import open3d as o3d


def _read_xyz_with_header_skip(xyz_path: str) -> o3d.geometry.PointCloud:

    pts = []
    cols6 = False

    with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except Exception:
                continue

            if len(parts) >= 6:
                try:
                    r = float(parts[3]); g = float(parts[4]); b = float(parts[5])
                    pts.append([x, y, z, r, g, b])
                    cols6 = True
                except Exception:
                    pts.append([x, y, z])
            else:
                pts.append([x, y, z])

    if not pts:
        raise ValueError(f"No numeric xyz rows found: {xyz_path}")

    arr = np.array(pts, dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))

    if cols6 and arr.shape[1] >= 6:
        rgb = arr[:, 3:6]
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    return pcd


def convert_xyz_to_ply_folder(input_folder: str, output_folder: str) -> None:
    input_folder = os.path.realpath(input_folder)
    output_folder = os.path.realpath(output_folder)

    for dirpath, _, filenames in os.walk(input_folder):
        xyz_files = [f for f in filenames if f.lower().endswith(".xyz")]
        if not xyz_files:
            continue

        rel_dir = os.path.relpath(dirpath, input_folder)
        out_dir = os.path.join(output_folder, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        for xyz_file in xyz_files:
            xyz_path = os.path.join(dirpath, xyz_file)
            ply_filename = os.path.splitext(xyz_file)[0] + ".ply"
            ply_path = os.path.join(out_dir, ply_filename)

            try:
                pcd = o3d.io.read_point_cloud(xyz_path, format="xyz")
                if len(pcd.points) == 0:
                    pcd = _read_xyz_with_header_skip(xyz_path)

                ok = o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)
                if not ok:
                    print(f"[FAIL] write failed: {ply_path}")
                    continue

                print(f"Converted '{xyz_path}' -> '{ply_path}'")

            except Exception as e:
                print(f"[ERROR] {xyz_path} : {e}")


if __name__ == "__main__":
    # TODO: Update these paths for your environment
    src_root  = "/path/to/..."
    save_root = "/path/to/..."
    convert_xyz_to_ply_folder(input_folder, output_folder)


