import sys
import argparse
import torch
import os
import numpy as np
import json
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Utils
from dynamic_models import GCN, NGFFobj, PointTransformer
from utils.transformation_utils import euler_xyz_to_matrix
from utils.general_utils import setup_seed
from dynamic_models.DGSDataset import LazyDGSDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint (.pth).")
    parser.add_argument("--input_h5", type=str, required=True, help="Input .h5 file containing the initial frame (e.g., 0000.h5).")
    parser.add_argument("--output_h5", type=str, required=True, help="Output .h5 file to store the predicted trajectory.")
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    setup_seed(seed=args.seed)

    # Validate paths
    if not os.path.exists(args.input_h5):
        raise FileNotFoundError(f"Input file not found: {args.input_h5}")
    
    out_dir = os.path.dirname(args.output_h5)
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #################################
    #   Set loading config path     #
    #################################
    dynamic_model_path = args.model_path
    
    test_steps = args.steps
    with open(os.path.join(dynamic_model_path.split('ngff_best.pth')[0], 'args.json'), 'r') as f:
        args_json = json.load(f)
    
    # Store old args that shouldn't be overwritten
    input_h5 = args.input_h5
    output_h5 = args.output_h5
    seed = args.seed
    
    args.__dict__.update(args_json)
    
    # Restore important args
    args.input_h5 = input_h5
    args.output_h5 = output_h5
    args.seed = seed
    args.k = 1
    args.steps = test_steps
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #################################
    #   Loading initial H5 data     #
    #################################
    seq_dir = os.path.dirname(args.input_h5)
    print(f"Loading data from sequence directory {seq_dir} using LazyDGSDataset...")
    
    # We load just 1 frame (the initial state)
    dataset = LazyDGSDataset(sequence_dirs=[seq_dir], 
                             num_frames=1, 
                             num_keypoints=args.num_keypoints, 
                             k=args.k, 
                             chunk=1, 
                             dtype=torch.float32, 
                             device=device,
                             cache=False)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data = next(iter(dataloader))
    
    # Data from dataloader is already batched (B=1)
    points = data['points'].to(device)
    com = data['com'].to(device)
    com_vel = data['com_vel'].to(device)
    angle = data['angle'].to(device)
    angle_vel = data['angle_vel'].to(device)
    padding = data['padding'].to(device)
    knn = data['knn'].to(device) if 'knn' in data else None
    
    #################################
    # NGFF simulation on points     #
    #################################
    print(f"Initializing {args.dynamic_model} model...")
    if args.dynamic_model == 'ngff':
        model = NGFFobj(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_keypoints=args.num_keypoints, k=args.k, mass=args.mass, 
                        dt=args.dt, ode_method=args.ode_method, r=0.1, step_size=args.step_size, threshold=args.threshold, rtol=args.rtol, atol=args.atol)
    elif args.dynamic_model == 'pointformer':
        model = PointTransformer(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers)
    elif args.dynamic_model == 'gcn':
        model = GCN(input_dim=args.output_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_layers=args.num_layers, r=0.1)
    else:
        raise ValueError(f"Unknown dynamic model: {args.dynamic_model}")
    
    state_dict = torch.load(dynamic_model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print(f"Running simulation for {args.steps} steps...")
    with torch.no_grad():
        if args.dynamic_model == 'ngff':
            point_seq, com_seq, angle_seq = model(points[:, 0], com[:, 0], com_vel[:, 0],
                                                  angle[:, 0], angle_vel[:, 0], padding, knn,
                                                  pred_len=args.steps-1, external_forces=None)
            
            # compute point trajectory using translation, rotation, and deformation
            R_seq = euler_xyz_to_matrix(angle_seq)  # (B, T, num_objs, 3, 3)
            future_seq = torch.matmul(point_seq, R_seq.transpose(-2, -1)) + com_seq.unsqueeze(-2)  # (B, T, num_objs, N, 3)
        
        elif args.dynamic_model in ['pointformer', 'gcn'] :
            future_seq = model(points[:, 0], com[:, 0], com_vel[:, 0],
                               angle[:, 0], angle_vel[:, 0], padding, knn,
                               pred_len=args.steps-1, external_forces=None)
            # future_seq shape: (B, T, num_objs, N, 3)
    
    B, T, num_objs, N, C = future_seq.shape
    # Flatten object and points dimension to store nicely
    future_seq_flat = future_seq[0].reshape(T, num_objs * N, C).cpu().numpy()

    #################################
    #   Write Output H5             #
    #################################
    print(f"Writing predicted trajectory to {args.output_h5}...")
    with h5py.File(args.output_h5, 'w') as f:
        # We save 'points' of shape (T, num_objs * N, 3)
        f.create_dataset('points', data=future_seq_flat, dtype=np.float32)
        
        # Save starting inputs
        f.create_dataset('linear_velocities', data=com_vel[0, 0].cpu().numpy(), dtype=np.float32)
        f.create_dataset('angular_velocities', data=angle_vel[0, 0].cpu().numpy(), dtype=np.float32)
        
        # Add metadata attributes
        f.attrs['num_frames'] = T
        f.attrs['num_objs'] = num_objs
        f.attrs['num_keypoints'] = N

    print("Done!")