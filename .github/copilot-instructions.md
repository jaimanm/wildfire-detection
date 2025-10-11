# GitHub Copilot Instructions for Wildfire Detection

This repository contains a PyTorch-based wildfire detection system using satellite imagery from the Sen2Fire dataset. The project implements a U-Net model for semantic segmentation of fire regions in Sentinel-2 satellite images.

## Project Overview

This is a machine learning project focused on wildfire detection using multi-spectral satellite imagery. The system uses various band combinations from Sentinel-2 data including RGB, SWIR, NBR (Normalized Burn Ratio), and NDVI (Normalized Difference Vegetation Index) to detect fire areas.

## Code Structure

- **`model/`**: Neural network architectures (U-Net with quantum components)
  - `Networks.py`: Contains model definitions including DoubleConv, Down, Up modules, and quantum layers
- **`dataset/`**: Data loading and preprocessing
  - `Sen2Fire_Dataset.py`: Custom PyTorch dataset for Sen2Fire data
  - `train.txt`, `val.txt`, `test.txt`: Data split files
- **`utils/`**: Utility functions
  - `tools.py`: Helper functions for metrics and evaluation
- **`Train.py`** and **`train.py`**: Training scripts (script vs notebook-converted versions)
- **`test_all.py`**: Testing and inference script
- **`train.ipynb`**: Jupyter notebook for interactive training
- **Slurm scripts**: `submit.sh`, `submit_training.sh`, `submit_test.sh` for HPC job submission

## Coding Conventions

### Python Style
- Use 4-space indentation (follow PEP 8)
- Import statements organized by: standard library, third-party, local modules
- Type hints are used sparingly; follow existing patterns
- Comments use `#` for single-line explanations

### PyTorch Conventions
- Models inherit from `nn.Module` and implement `forward()` method
- Use `nn.Sequential` for simple layer stacking
- Prefer in-place operations where appropriate (`inplace=True`)
- Device management: check for CUDA availability and move tensors explicitly
- Random seeds: Use `init_seeds()` function for reproducibility

### Data Processing
- Sen2Fire dataset uses `.npz` files with 'image' and 'label' keys
- Input modes (0-11) select different band combinations:
  - 0: all_bands (12 channels)
  - 1: all_bands_aerosol (13 channels)
  - 2: rgb (3 channels)
  - 3: rgb_aerosol (4 channels)
  - 4: swir (3 channels)
  - 5: swir_aerosol (4 channels)
  - 6: nbr (3 channels)
  - 7: nbr_aerosol (4 channels)
  - 8: ndvi (3 channels)
  - 9: ndvi_aerosol (4 channels)
  - 10: rgb_swir_nbr_ndvi (various)
  - 11: rgb_swir_nbr_ndvi_aerosol (various)

### Training Patterns
- Use `tqdm` for progress bars in training loops
- Model checkpoints saved as `.pth` files
- Logging to text files in snapshot directories
- Training history stored as `.npz` files
- Metrics: Overall Accuracy (OA), mean IOU (mIOU), Precision, Recall, F1-score

### HPC/Slurm Integration
- Scripts are designed for GPU clusters with Slurm
- Environment modules: cuda, cudnn, pytorch
- Working directory typically set to `$HOME/scratch/wildfires`
- Jupyter notebooks converted to Python scripts before execution
- Job output directed to `jobs/slurm.$SLURM_JOB_ID/` directories

## Key Design Patterns

1. **Argument Parsing**: Use `argparse` with descriptive help text
2. **Data Loaders**: PyTorch `DataLoader` with configurable batch size and workers
3. **Model Initialization**: Check input mode to determine number of channels
4. **Training Loop**: 
   - Epoch-based with batch iteration
   - Manual gradient zeroing, loss computation, backward pass, optimizer step
   - Periodic validation and checkpoint saving
5. **Testing**: Scene-based reconstruction from patches with overlap handling

## Common Operations

### Adding New Input Modes
When adding a new band combination:
1. Update `modename` list in `dataset/Sen2Fire_Dataset.py`
2. Add corresponding composite creation function
3. Update model instantiation logic to handle new channel count
4. Update argument parser help text

### Modifying the Model
- Model changes go in `model/Networks.py`
- Update channel handling in `Train.py` when changing architecture
- Consider bilinear vs. transpose convolution for upsampling

### Adjusting Training
- Hyperparameters in `get_arguments()` function
- Loss function: weighted cross-entropy (adjust `--weight` parameter)
- Optimizer: Adam with configurable learning rate
- Early stopping based on F1 score

## Testing and Validation

- Run training scripts with appropriate `--mode` and `--data_dir` arguments
- Test scripts expect trained model checkpoint via `--restore_from`
- Output maps saved to `--snapshot_dir`
- Scene-based evaluation with dimensions specified in `scene_dims` dict

## Important Notes

- **Environment**: Code assumes HPC environment with Slurm workload manager
- **Data**: Sen2Fire dataset must be available at specified `--data_dir`
- **GPU**: Training requires CUDA-capable GPU (A100 preferred)
- **Dependencies**: PyTorch, numpy, tqdm, matplotlib, scipy, pennylane (for quantum components)
- **Quantum Components**: Some experimental quantum layers present but may be commented out

## When Making Changes

- Maintain compatibility with both script (`Train.py`) and notebook (`train.ipynb`) versions
- Update Slurm submission scripts if changing file names or execution patterns
- Preserve data loading patterns to ensure compatibility with Sen2Fire format
- Test with small number of epochs/batches before full training runs
- Consider GPU memory constraints when modifying batch sizes or model architecture
