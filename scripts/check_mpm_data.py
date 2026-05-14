import os
import argparse
import shutil

def check_folder(folder_path):
    # Define required files
    required_files = {"opacity.h5", "shs.h5"}
    for i in range(100):
        required_files.add(f"{i:04d}.h5")
        
    # Check if all required files exist
    for f in required_files:
        if not os.path.exists(os.path.join(folder_path, f)):
            return False # Missing at least one file
            
    return True # Complete

def main():
    parser = argparse.ArgumentParser(description="Check MPM dataset folders for completeness.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the base mpm directory (e.g., /path/to/mpm)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    incomplete_folders = []
    total_folders = 0

    # The structure is base_dir/<group>/<scene>/
    # Iterate through groups (e.g., 3_0, 3_1, etc.)
    for group in os.listdir(base_dir):
        group_path = os.path.join(base_dir, group)
        if os.path.isdir(group_path) and not group.startswith('.'):
            # Iterate through scenes (e.g., 0_panda_can)
            for scene in os.listdir(group_path):
                scene_path = os.path.join(group_path, scene)
                if os.path.isdir(scene_path) and not scene.startswith('.'):
                    total_folders += 1
                    if not check_folder(scene_path):
                        incomplete_folders.append(scene_path)

    print(f"Checked {total_folders} scene folders.")
    if incomplete_folders:
        print(f"Found {len(incomplete_folders)} incomplete folders. Deleting them...")
        for folder in incomplete_folders:
            print(f"Deleting: {folder}")
            shutil.rmtree(folder)
    else:
        print("All folders are complete!")

if __name__ == "__main__":
    main()

