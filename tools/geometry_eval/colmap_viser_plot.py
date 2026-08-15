import argparse
import time
import json
import datetime
from pathlib import Path
import numpy as np
import viser
import viser.transforms as tf
import PIL.Image as Image


def read_colmap_text_data(path: Path):
    """Read data directly from the .txt files without pycolmap."""
    points3d = {}
    # Read points3D.txt
    with open(path / "points3D.txt", "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            elems = line.split()
            id = int(elems[0])
            xyz = np.array([float(elems[1]), float(elems[2]), float(elems[3])])
            rgb = np.array([int(elems[4]), int(elems[5]), int(elems[6])])
            points3d[id] = (xyz, rgb)

    images = []
    # Read images.txt
    with open(path / "images.txt", "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            line = lines[i].strip()
            if line.startswith("#") or not line: continue
            elems = line.split()
            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz = map(float, elems[5:8])
            name = elems[9]
            images.append({
                "name": name,
                "quat": [qw, qx, qy, qz], # wxyz
                "trans": [tx, ty, tz]
            })

    return points3d, images

def visualize_colmap_fallback(dataset_path: str, input_views: set = None, sync_json_paths: list = None):
    server = viser.ViserServer()
    path = Path(dataset_path)

    server.gui.configure_theme(
        dark_mode=True,
    )

    # Screenshot save path setup
    screenshot_dir = path / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    print("Reading text files directly...")
    points3d, images = read_colmap_text_data(path)

    # 1. Plot the point cloud
    xyz_list = np.array([v[0] for v in points3d.values()])
    rgb_list = np.array([v[1] for v in points3d.values()])

    server.scene.add_point_cloud(
        "/points3D",
        points=xyz_list,
        colors=rgb_list,
        point_size=0.02,
    )

    # 3. Plot the cameras
    # input_views is a set of basenames (e.g. 000.png)
    # names in images.txt may include subdirectories (e.g. images/000.png), so compare by basename
    if input_views is not None:
        sample_img_names = [img['name'] for img in images[:3]]
        sample_basenames = [Path(n).name for n in sample_img_names]
        print(f"[debug] images.txt sample names: {sample_img_names}")
        print(f"[debug] images.txt sample basenames: {sample_basenames}")
        print(f"[debug] input_views sample: {sorted(input_views)[:3]}")
        matched = sum(1 for img in images if Path(img['name']).name in input_views)
        print(f"[debug] matched input views: {matched}/{len(images)}")
    else:
        print("[warning] input_views not found. All cameras will be shown in blue.")

    camera_centers = []
    camera_scale = 0.05
    for img in images:
        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(np.array(img["quat"])),
            np.array(img["trans"])
        ).inverse()

        center = T_world_camera.translation()
        camera_centers.append(center)

        # input view: red, others: blue
        # names in images.txt may have a subdirectory prefix, so compare by basename
        img_name = img['name']
        img_basename = Path(img_name).name
        if input_views is not None and img_basename in input_views:
            cam_color = (255, 0, 0)   # red (input view)
        else:
            cam_color = (0, 255, 255)   # cyan (non-input / novel view)

        server.scene.add_camera_frustum(
            f"/cameras/{img_name}",
            fov=0.8,
            aspect=1.0,
            scale=camera_scale,
            wxyz=T_world_camera.rotation().wxyz,
            position=center,
            color=cam_color,
        )

    # 3. Add camera-center lines (trajectory)
    if len(camera_centers) > 1:
        camera_centers = np.array(camera_centers)

        # Build start/end point pairs for the line segments, shaped (N-1, 2, 3)
        points_start = camera_centers[:-1]
        points_end = camera_centers[1:]

        # Stack into shape (N-1, 2, 3)
        line_vertices = np.stack([points_start, points_end], axis=1) # Shape: (N-1, 2, 3)

        server.scene.add_line_segments(
            "/trajectory",
            points=line_vertices, # Changed: pass in (N, 2, 3) form
            # Colors may need to be (N, 2, 3) or (N, 3); safely use a single color or tiling
            colors=np.array([255, 0, 0]),
            line_width=4.0,
        )

    # --- Viewpoint save/load + screenshots ---

    with server.gui.add_folder("Viewpoint Control"):
        save_btn = server.gui.add_button("Save Current View", icon=viser.Icon.CAMERA)
        load_btn = server.gui.add_button("Load Last View", icon=viser.Icon.DOWNLOAD)
        status_text = server.gui.add_text("Status", initial_value="Ready", disabled=True)

    with server.gui.add_folder("Screenshot"):
        width_slider  = server.gui.add_slider("Width",  min=256, max=3840, step=1, initial_value=1920)
        height_slider = server.gui.add_slider("Height", min=256, max=2160, step=1, initial_value=1080)
        capture_btn   = server.gui.add_button("Capture Current View (JPEG)", icon=viser.Icon.SCREENSHOT)
        capture_text  = server.gui.add_text("Saved", initial_value="", disabled=True)

    json_path = path / "viser_viewpoint.json"

    # sync_json_paths: list of paths in other vipe folders of the same scene to save to as well
    all_save_paths = [json_path]
    if sync_json_paths:
        for p in sync_json_paths:
            p = Path(p)
            if p != json_path and p not in all_save_paths:
                all_save_paths.append(p)

    @server.on_client_connect
    def setup_viewpoint_handlers(client: viser.ClientHandle) -> None:

        @save_btn.on_click
        def _(_) -> None:
            view_data = {
                "position": client.camera.position.tolist(),
                "wxyz": client.camera.wxyz.tolist(),
                "look_at": client.camera.look_at.tolist(),
                "fov": client.camera.fov,
                "aspect": client.camera.aspect,
            }
            saved_to = []
            for save_path in all_save_paths:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(view_data, f, indent=4)
                saved_to.append(save_path.parent.name)
                print(f"Viewpoint saved: {save_path}")
            status_text.value = f"Saved to: {', '.join(saved_to)}"

        @load_btn.on_click
        def _(_) -> None:
            if not json_path.exists():
                status_text.value = "No saved view found!"
                return
            with open(json_path, "r") as f:
                data = json.load(f)
            client.camera.position = np.array(data["position"])
            client.camera.wxyz = np.array(data["wxyz"])
            client.camera.look_at = np.array(data["look_at"])
            client.camera.fov = data["fov"]
            status_text.value = "View loaded!"
            print("Viewpoint loaded and applied.")

        @capture_btn.on_click
        def _(_) -> None:
            W = int(width_slider.value)
            H = int(height_slider.value)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Save the camera viewpoint first before capturing (in case the connection drops)
            view_data = {
                "position": client.camera.position.tolist(),
                "wxyz": client.camera.wxyz.tolist(),
                "look_at": client.camera.look_at.tolist(),
                "fov": client.camera.fov,
                "aspect": client.camera.aspect,
            }
            for save_path in all_save_paths:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(view_data, f, indent=4)
            vp_path = screenshot_dir / f"viewpoint_{timestamp}.json"
            with open(vp_path, "w") as f:
                json.dump(view_data, f, indent=4)
            print(f"Viewpoint pre-saved: {vp_path}")
            try:
                capture_text.value = "Capturing... do not close the tab"
                img = client.get_render(height=H, width=W, transport_format="jpeg")
                save_path = screenshot_dir / f"screenshot_{timestamp}.jpg"
                Image.fromarray(img).save(save_path)
                capture_text.value = save_path.name
                print(f"Screenshot saved: {save_path}")
            except Exception as e:
                capture_text.value = f"ERROR: {e}"
                print(f"Screenshot failed (viewpoint was saved): {e}")

    # -----------------------------------------------

    print(f"Visualization ready! Connect at: {server.get_host()}")
    while True:
        time.sleep(1)

def find_input_views(base_dir: str, dataset: str, scene: str) -> set:
    """
    Automatically look up the input view list in the folder whose name is base_dir
    with the '_vipe' suffix removed.
    e.g. ours_vipe -> ours,  eccv_ours_vipe -> eccv_ours
    Path: {ours_folder}/{dataset}/img2vid/{scene}/input/
    Both dataset folder-name forms 'dtu_3_1' / 'dtu_3_1.0' are tried.
    """
    base_path = Path(base_dir)
    ours_folder_name = base_path.name.replace("_vipe", "")
    ours_folder = base_path.parent / ours_folder_name

    # Dataset folder-name candidates: original, with .0 appended, with .0 removed
    dataset_candidates = [dataset]
    if dataset.endswith(".0"):
        dataset_candidates.append(dataset[:-2])  # remove .0
    else:
        dataset_candidates.append(dataset + ".0")  # append .0

    for ds in dataset_candidates:
        input_dir = ours_folder / ds / "img2vid" / scene / "input"
        if input_dir.exists():
            input_views = set(p.name for p in input_dir.iterdir() if p.is_file())
            print(f"Input views auto-detected: {len(input_views)} (red) [{input_dir}]")
            return input_views

    print(f"Input view folder not found: {ours_folder}/{dataset_candidates}/img2vid/{scene}/input")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP results with viser.")
    parser.add_argument("--base_dir", type=str, default="./ours_vipe",
                        help="Dataset root directory (default: ./ours_vipe). "
                             "Input views are auto-detected in the folder with the _vipe suffix removed.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g. dl3dv10_6_1)")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name (e.g. e78f8cebd2bd93d960bfa...)")
    parser.add_argument("--input_views_file", type=str, default=None,
                        help="File listing input view image names (one per line). "
                             "Takes precedence over auto-detection when given.")
    parser.add_argument("--sync_vipe_names", type=str, nargs="+",
                        default=["gt_vipe", "eccv_ours_vipe", "eccv_seva_input_vipe"],
                        help="Sibling *_vipe folder names of the same scene to which the saved "
                             "viewpoint is synced (default: gt_vipe eccv_ours_vipe eccv_seva_input_vipe)")
    args = parser.parse_args()

    dataset_path = str(Path(args.base_dir) / args.dataset / f"{args.scene}_colmap")

    if args.input_views_file is not None:
        with open(args.input_views_file, "r") as f:
            input_views = set(line.strip() for line in f if line.strip())
        print(f"Input views loaded: {len(input_views)} (red) [{args.input_views_file}]")
    else:
        input_views = find_input_views(args.base_dir, args.dataset, args.scene)

    # Save the viewpoint simultaneously to gt_vipe, eccv_ours_vipe, eccv_seva_input_vipe of the same scene
    base_path = Path(args.base_dir)
    parent_dir = base_path.parent
    sync_vipe_names = args.sync_vipe_names
    sync_json_paths = []
    for vipe_name in sync_vipe_names:
        vipe_colmap_dir = parent_dir / vipe_name / args.dataset / f"{args.scene}_colmap"
        sync_json_paths.append(vipe_colmap_dir / "viser_viewpoint.json")
        print(f"Sync target: {vipe_colmap_dir / 'viser_viewpoint.json'}")

    visualize_colmap_fallback(dataset_path, input_views=input_views, sync_json_paths=sync_json_paths)
