import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import *
from dataset.Sen2Fire_Dataset import Sen2FireDataSet
from model.Networks import unet
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy import sparse

name_classes = np.array(['non-fire','fire'], dtype=str)
epsilon = 1e-14

scene_dims = {
    1: (32, 27),
    2: (22, 27),
    3: (14, 36),
    4: (21, 24)
}

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../Sen2Fire', help="dataset path.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt', help="test list file.")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes.")
    parser.add_argument("--mode", type=int, default=5, help="input type (0-all_bands, 1-all_bands_aerosol,...).")
    parser.add_argument("--batch_size", type=int, default=8, help="number of images in each batch.")
    parser.add_argument("--num_workers", type=int, default=int(os.environ['SLURM_CPUS_PER_TASK']), help="number of workers for multithread dataloading.")
    parser.add_argument("--restore_from", type=str, default='./Exp/swir_aerosol/weight_10_time0815_1330/best_model.pth', help="trained model.")
    parser.add_argument("--snapshot_dir", type=str, default='./Map/', help="where to save detection results.")
    return parser.parse_args()

modename = ['all_bands', 'all_bands_aerosol', 'rgb', 'rgb_aerosol', 'swir', 'swir_aerosol', 'nbr', 'nbr_aerosol', 'ndvi', 'ndvi_aerosol', 'rgb_swir_nbr_ndvi', 'rgb_swir_nbr_ndvi_aerosol']


import os

def process_scene(scene_id, args, snapshot_dir):
    device_id = getattr(args, 'device_id', None)
    if device_id is not None:
        print(f"Processing scene {scene_id} on device {device_id}")
    else:
        print(f"Processing scene {scene_id} on CPU")
    scene_list_files = {
        1: './dataset/train.txt',
        2: './dataset/train.txt',
        3: './dataset/val.txt',
        4: './dataset/test.txt',
    }
    scene_list = scene_list_files[scene_id]
    if not os.path.exists(scene_list):
        print(f"List file {scene_list} not found, skipping scene {scene_id}.")
        return
    input_size = (512, 512)
    # Set device to CUDA if available, else CPU
    use_cuda = torch.cuda.is_available() and (device_id is not None)
    if use_cuda:
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    # Create network
    if args.mode == 0:
        model = unet(n_classes=args.num_classes, n_channels=12)
    elif args.mode == 1:
        model = unet(n_classes=args.num_classes, n_channels=13)
    elif args.mode in [2,4,6,8]:
        model = unet(n_classes=args.num_classes, n_channels=3)
    elif args.mode in [3,5,7,9]:
        model = unet(n_classes=args.num_classes, n_channels=4)
    elif args.mode == 10:
        model = unet(n_classes=args.num_classes, n_channels=6)
    elif args.mode == 11:
        model = unet(n_classes=args.num_classes, n_channels=7)
    # Explicitly set weights_only=True to avoid FutureWarning and for security
    saved_state_dict = torch.load(args.restore_from, map_location=device, weights_only=True)
    model.load_state_dict(saved_state_dict)
    model = model.to(device)
    model.eval()
    test_loader = data.DataLoader(
        Sen2FireDataSet(args.data_dir, scene_list, mode=args.mode),
        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear').to(device)
    preds_dir = os.path.join(snapshot_dir, 'preds', f'Scene{scene_id}')
    os.makedirs(preds_dir, exist_ok=True)
    for _, batch in enumerate(tqdm(test_loader, desc=f"Generating predictions for scene {scene_id}")):
        image, _,_,name = batch
        image = image.float().to(device)
        patch_name = name[0].split('/')[1]
        filename = patch_name[:6] + str(scene_id) + patch_name[7:]
        patch_path = os.path.join(preds_dir, filename)
        # Skip prediction if file already exists
        if os.path.exists(patch_path + '.npz'):
            continue
        with torch.no_grad():
            pred = model(image)
        _,pred = torch.max(interp(nn.functional.softmax(pred,dim=1)).detach(), 1)
        pred = pred.squeeze().data.cpu().numpy().astype('uint8')
        # Save prediction as sparse matrix
        sparse_pred = sparse.csr_matrix(pred)
        sparse.save_npz(patch_path, sparse_pred)
    # Reconstruct map
    image_dir = os.path.join(args.data_dir, f"scene{scene_id}")
    n_row, n_col = scene_dims[scene_id]
    patch_size = (512, 512)
    overlap = 128
    original_image_size = (3, n_row * (patch_size[0] - overlap) + overlap, n_col * (patch_size[1] - overlap) + overlap)
    reconstructed_rgb = np.zeros(original_image_size)
    reconstructed_label = np.zeros(original_image_size[1:])
    reconstructed_pred = np.zeros(original_image_size[1:])
    preds_dir = os.path.join(snapshot_dir, 'preds', f'Scene{scene_id}')
    for row in tqdm(range(1, n_row + 1), desc=f"Building map for scene {scene_id}"):
        for col in range(1, n_col + 1):
            patch_name = f"scene_{scene_id}_patch_{row}_{col}.npz"
            patch_path = os.path.join(image_dir, patch_name)
            patch_data = np.load(patch_path)['image']
            patch_gt = np.load(patch_path)['label']
            filename = patch_name[:6] + str(scene_id) + patch_name[7:]
            pred_path = os.path.join(preds_dir, filename)
            patch_pred = sparse.load_npz(pred_path).toarray()
            start_row = (row - 1) * (patch_size[0] - overlap)
            start_col = (col - 1) * (patch_size[1] - overlap)
            reconstructed_rgb[:, start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_data[[3,2,1],:,:]
            reconstructed_label[start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_gt
            if (col==0)and(row==0):
                reconstructed_pred[start_row:start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_pred
            elif (col==0)and(row!=0):
                reconstructed_pred[start_row+int(overlap/2):start_row + patch_size[0], start_col:start_col + patch_size[1]] = patch_pred[int(overlap/2):,:]
            elif (col!=0)and(row==0):
                reconstructed_pred[start_row:start_row + patch_size[0], start_col+int(overlap/2):start_col + patch_size[1]] = patch_pred[:,int(overlap/2):]
            elif (col!=0)and(row!=0):
                reconstructed_pred[start_row+int(overlap/2):start_row + patch_size[0], start_col+int(overlap/2):start_col + patch_size[1]] = patch_pred[int(overlap/2):,int(overlap/2):]
    plt.style.use('ggplot')
    cmap = ListedColormap(['white', 'red'])
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500.)
    axs[0].axis('off')
    axs[0].set_title('RGB image', fontsize=12)
    axs[1].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500., alpha=0.6)
    axs[1].imshow(reconstructed_pred, cmap=cmap, alpha=0.7)
    axs[1].axis('off')
    axs[1].set_title('Detection', fontsize=12)
    axs[2].imshow(reconstructed_rgb.transpose(1, 2, 0)/1500., alpha=0.6)
    axs[2].imshow(reconstructed_label, cmap=cmap, alpha=0.7)
    axs[2].axis('off')
    axs[2].set_title('Label', fontsize=12)
    legend_labels = ['Non-fire', 'Fire']
    plt.legend(handles=[Rectangle((0,0),1,1,facecolor=cmap(i),edgecolor='black') for i in range(2)], labels=legend_labels, fontsize=12, frameon=False, bbox_to_anchor=(1.04, 0), loc="lower left")
    plt.tight_layout()
    save_path = os.path.join(snapshot_dir, f"scene{scene_id}_map.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    args = get_arguments()
    mode_name = modename[args.mode]
    weight_dir = os.path.basename(os.path.dirname(args.restore_from))
    out_dir = os.path.join(args.snapshot_dir, mode_name, weight_dir)
    if os.path.exists(out_dir):
        print(f"Output directory {out_dir} already exists. Exiting.")
        return
    os.makedirs(out_dir, exist_ok=True)
    scene_ids = [1, 2, 3, 4]
    # Set num_workers to all available CPUs if not set by user
    if args.num_workers is None or args.num_workers < 1:
        args.num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
    # Process scenes sequentially for best GPU utilization
    for scene_id in scene_ids:
        args.device_id = 0 if torch.cuda.is_available() else None
        process_scene(scene_id, args, out_dir)

if __name__ == '__main__':
    main()
