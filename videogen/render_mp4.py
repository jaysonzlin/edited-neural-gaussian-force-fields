import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os
import argparse
import h5py

import imageio
from PIL import Image
from io import BytesIO

def camera_view_dir_y(elev, azim):
    """Unit vector for camera direction with Y as 'up'."""
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    dx = np.sin(azim_rad) * np.cos(elev_rad)
    dy = np.sin(elev_rad)
    dz = np.cos(azim_rad) * np.cos(elev_rad)
    return np.array([dx, dy, dz])

def compute_depth(points, elev, azim):
    """Project points onto the camera's view direction (Y as 'up')."""
    view_dir = camera_view_dir_y(elev, azim)
    # depth = p · view_dir
    depth = points @ view_dir
    return depth

def save_pointcloud_video_genesis(points, drag_points, save_path, fps=48, point_color='blue', grid_lim=4, vertical_axis='z',
    elev=10, azim=45, floor_height=None, show_trajectory=False, trajectory_len=5):
    
    # Configure the figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Compute dynamic bounding box
    points_flat = points.reshape(-1, 3)
    min_x, min_y, min_z = np.min(points_flat, axis=0)
    max_x, max_y, max_z = np.max(points_flat, axis=0)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    # Isotropic padding
    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2
    pad = max_range * 1.2
    
    if floor_height is None:
        floor_height = min_y if vertical_axis == 'y' else min_z

    # We handle vertical_axis manually to ensure compatibility with all matplotlib versions
    ax.view_init(elev=elev, azim=azim)
    
    # Plot and save each frame
    cmap_1 = plt.colormaps.get_cmap('cool')
    cmap_2 = plt.colormaps.get_cmap('autumn')
    frames = []
    
    for i in range(points.shape[0]):
        
        frame_points = points[i]
        depth_frame_points = compute_depth(frame_points, elev=elev, azim=azim)
        
        # If the point count is divisible by 2048, we assume it's pre-processed objects and normalize per-object
        # Otherwise, for raw MPM files with variable point counts, we do global depth normalization
        n_pts_per_obj = 2048
        color_frame_points = np.zeros((frame_points.shape[0], 4))
        
        if frame_points.shape[0] % n_pts_per_obj == 0:
            for obj_idx in range(0, frame_points.shape[0], n_pts_per_obj):
                obj_depths = depth_frame_points[obj_idx : obj_idx + n_pts_per_obj]
                d_min, d_max = obj_depths.min(), obj_depths.max()
                if d_max - d_min < 1e-6:
                    obj_norm = np.zeros_like(obj_depths)
                else:
                    obj_norm = (obj_depths - d_min) / (d_max - d_min)
                color_frame_points[obj_idx : obj_idx + n_pts_per_obj] = cmap_1(obj_norm)
        else:
            d_min, d_max = depth_frame_points.min(), depth_frame_points.max()
            if d_max - d_min < 1e-6:
                global_norm = np.zeros_like(depth_frame_points)
            else:
                global_norm = (depth_frame_points - d_min) / (d_max - d_min)
            color_frame_points = cmap_1(global_norm)
        if len(drag_points) > 0:
            frame_points_drag = drag_points[i]
            depth_frame_points_drag = compute_depth(frame_points_drag, elev=elev, azim=azim)
            depth_frame_points_drag_normalized = (depth_frame_points_drag - depth_frame_points_drag.min()) / \
                (depth_frame_points_drag.max() - depth_frame_points_drag.min())
            color_frame_points_drag = cmap_2(np.ones_like(depth_frame_points_drag_normalized) * -10)
            all_points = np.concatenate([frame_points, frame_points_drag], axis=0)
            all_color = np.concatenate([color_frame_points, color_frame_points_drag], axis=0)
        else:
            all_points, all_color = frame_points, color_frame_points
            
        ax.clear()
        
        fh = floor_height if floor_height is not None else 0.0
        
        # Draw tiny axes placed on the floor plane
        if vertical_axis == 'y':
            # Data Y maps to Screen Z, Data Z maps to -Screen Y
            ax.plot([0, 0.5], [0, 0], [fh, fh], color='red', linewidth=1, alpha=0.8, zorder=10)   # X
            ax.plot([0, 0], [0, 0], [fh, fh + 0.5], color='green', linewidth=1, alpha=0.8, zorder=10) # Y -> up
            ax.plot([0, 0], [0, -0.5], [fh, fh], color='blue', linewidth=1, alpha=0.8, zorder=10)  # Z -> forward
        else:
            # Z is up
            ax.plot([0, 0.5], [0, 0], [fh, fh], color='red', linewidth=1, alpha=0.8, zorder=10)   # X
            ax.plot([0, 0], [0, 0.5], [fh, fh], color='green', linewidth=1, alpha=0.8, zorder=10) # Y
            ax.plot([0, 0], [0, 0], [fh, fh + 0.5], color='blue', linewidth=1, alpha=0.8, zorder=10)  # Z
        
        # Reference dots for a 4-meter total span (-2 to 2)
        grid_vals = np.arange(-2, 2.1, 0.25)
        if vertical_axis == 'y':
            ax.scatter(grid_vals, np.zeros_like(grid_vals), np.full_like(grid_vals, fh), color='red', s=2, alpha=0.5, zorder=5)
            ax.scatter(np.zeros_like(grid_vals), np.zeros_like(grid_vals), grid_vals + fh, color='green', s=2, alpha=0.5, zorder=5)
            ax.scatter(np.zeros_like(grid_vals), -grid_vals, np.full_like(grid_vals, fh), color='blue', s=2, alpha=0.5, zorder=5)
        else:
            ax.scatter(grid_vals, np.zeros_like(grid_vals), np.full_like(grid_vals, fh), color='red', s=2, alpha=0.5, zorder=5)
            ax.scatter(np.zeros_like(grid_vals), grid_vals, np.full_like(grid_vals, fh), color='green', s=2, alpha=0.5, zorder=5)
            ax.scatter(np.zeros_like(grid_vals), np.zeros_like(grid_vals), grid_vals + fh, color='blue', s=2, alpha=0.5, zorder=5)
        
        # Draw floor if specified
        if floor_height is not None:
            # Create a large plane for the floor
            floor_size = pad * 2
            # Use a grid for the gradient
            c = np.linspace(-floor_size, floor_size, 10) # Low res is fine for solid color
            C1, C2 = np.meshgrid(c, c)
            
            # Since we manually map coordinates, floor is always plotted at Screen Z = floor_height
            if vertical_axis == 'y':
                X = C1 + center_x
                Y = C2 - center_z # Data Z maps to -Screen Y
            else:
                X = C1 + center_x
                Y = C2 + center_y
            Z = np.full_like(C1, floor_height)
            ax.plot_surface(X, Y, Z, color='red', alpha=0.1, zorder=0, shade=False, antialiased=False, linewidth=0)

        if show_trajectory:
            t_start = max(0, i - trajectory_len)
            if i > t_start:
                # Use Line3DCollection for efficient line plotting
                # Shape of segments: (N_points, L_trajectory, 3)
                segments = points[t_start:i+1].transpose(1, 0, 2)
                line_collection = Line3DCollection(segments, colors=color_frame_points, 
                                                   linewidths=0.5, alpha=0.01, zorder=1)
                ax.add_collection3d(line_collection)

        ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
            c=all_color, s=1, depthshade=False, alpha=0.9)
             
        ax.axis('off')  # Turn off the axes
        ax.grid(False)  # Hide the grid
        
        # Zoomed in on the action
        if vertical_axis == 'y':
            # Data X -> Screen X
            ax.set_xlim(center_x - pad, center_x + pad)
            # Data Z -> Screen -Y
            ax.set_ylim(-center_z - pad, -center_z + pad)
            # Data Y -> Screen Z (Height). We want the floor at the bottom.
            ax.set_zlim(floor_height, floor_height + 2 * pad)
        else:
            # Data X -> Screen X
            ax.set_xlim(center_x - pad, center_x + pad)
            # Data Y -> Screen Y
            ax.set_ylim(center_y - pad, center_y + pad)
            # Data Z -> Screen Z (Height). Floor at bottom.
            ax.set_zlim(floor_height, floor_height + 2 * pad)

        # Set box aspect dynamically
        ax.set_box_aspect([1, 1, 1])
        
        # Adjust margins for tight layout
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save frame
        buf = BytesIO()
        plt.savefig(buf, bbox_inches='tight', pad_inches=0.0, dpi=300)
        buf.seek(0)
        frames.append(np.array(Image.open(buf)))

    plt.close()
    
    imageio.mimsave(save_path, frames, fps=fps, macro_block_size=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an MP4 video from an H5 file containing predicted points.")
    parser.add_argument("--input_h5", type=str, required=True, help="Input .h5 file containing predicted points, or a directory containing a sequence of .h5 files.")
    parser.add_argument("--output_mp4", type=str, required=True, help="Output .mp4 file to store the video.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video.")
    parser.add_argument("--vertical_axis", type=str, default='z', help="Vertical axis for plotting ('y' or 'z').")
    parser.add_argument("--elev", type=float, default=20, help="Elevation angle for the camera view.")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth angle for the camera view.")
    args = parser.parse_args()

    print(f"Loading data from {args.input_h5}...")
    
    if os.path.isdir(args.input_h5):
        # It's a directory containing a sequence of .h5 frames (like GSCollision dataset)
        files = sorted([f for f in os.listdir(args.input_h5) if f.endswith('.h5') and f[0].isdigit()])
        points_list = []
        for f_name in files:
            with h5py.File(os.path.join(args.input_h5, f_name), 'r') as f:
                # The MPM frame files use 'pos' instead of 'points'
                if 'pos' in f:
                    points_list.append(f['pos'][()])
                elif 'points' in f:
                    points_list.append(f['points'][()])
        points = np.stack(points_list, axis=0) # (T, N, 3)
    else:
        # It's a single compiled .h5 file (like the prediction output)
        with h5py.File(args.input_h5, 'r') as f:
            points = f['points'][()]
        
    print(f"Loaded points with shape: {points.shape}")
    
    print(f"Generating video and saving to {args.output_mp4}...")
    save_pointcloud_video_genesis(
        points=points,
        drag_points=[],
        save_path=args.output_mp4,
        fps=args.fps,
        vertical_axis=args.vertical_axis,
        elev=args.elev,
        azim=args.azim
    )
    print(f"Video successfully saved to {args.output_mp4}")