---
name: Local Environment Setup
about: Get the existing code running on a local machine
title: '[SETUP] '
labels: 'setup, documentation'
assignees: ''
---

## Objective
Set up the development environment and successfully run the wildfire detection model on a local machine.

## Setup Tasks

### Environment Setup
- [ ] Install Python (version: ___ )
- [ ] Create virtual environment (conda/venv/virtualenv)
- [ ] Install PyTorch
  - [ ] CPU version or GPU version (specify CUDA version)
  - [ ] Verify PyTorch installation
- [ ] Install PennyLane
- [ ] Install other dependencies:
  - [ ] NumPy
  - [ ] tqdm
  - [ ] matplotlib
  - [ ] scipy
  - [ ] Other packages from imports
- [ ] Document all dependency versions

### Dataset Setup
- [ ] Obtain access to Sen2Fire dataset
- [ ] Download dataset
- [ ] Verify dataset structure:
  - [ ] Check for .npz files
  - [ ] Verify 'image', 'aerosol', and 'label' arrays
  - [ ] Confirm dataset location matches code paths
- [ ] Verify train/val/test split files:
  - [ ] `dataset/train.txt`
  - [ ] `dataset/val.txt`
  - [ ] `dataset/test.txt`
- [ ] Update data paths in code if necessary

### Code Verification
- [ ] Clone/download repository
- [ ] Verify all required files are present
- [ ] Update hardcoded paths:
  - [ ] Data directory paths
  - [ ] Output directory paths
  - [ ] Any user-specific paths
- [ ] Check import statements work

### Training Test
- [ ] Run training with limited epochs (e.g., 1-2 epochs)
- [ ] Use small batch size if memory constrained
- [ ] Verify training starts without errors
- [ ] Monitor GPU/CPU usage
- [ ] Check that loss decreases
- [ ] Verify model checkpoints are saved
- [ ] Document training time per epoch

### Testing/Inference Test
- [ ] Load a trained model checkpoint
- [ ] Run inference on test set
- [ ] Verify predictions are generated
- [ ] Check output format and structure
- [ ] Verify evaluation metrics are calculated

### Bug Tracking
- [ ] Document any minor bugs found
- [ ] Fix minor bugs if possible:
  - [ ] Import errors
  - [ ] Path issues
  - [ ] Deprecated function warnings
  - [ ] Type mismatches
- [ ] Report major bugs separately (create new issues)
- [ ] Log all changes made

## Hardware Specifications
Document your setup:
- **OS**: 
- **CPU**: 
- **RAM**: 
- **GPU** (if applicable): 
- **CUDA Version** (if applicable): 
- **Storage Available**: 

## Software Versions
Document versions used:
- **Python**: 
- **PyTorch**: 
- **PennyLane**: 
- **CUDA/cuDNN** (if applicable): 
- **Other key packages**: 

## Configuration Used
- **Batch Size**: 
- **Number of Epochs**: 
- **Input Mode**: 
- **Training Set Size** (# of samples): 
- **Dataset Location**: 

## Performance Metrics
Document performance on your machine:
- **Training Time per Epoch**: 
- **Memory Usage (GPU/CPU)**: 
- **Disk Space Used**: 
- **Final Training Loss**: 
- **Validation Metrics** (if evaluated): 

## Optional: Setup Script
- [ ] Create an automated setup script
- [ ] Include dependency installation
- [ ] Include dataset download instructions
- [ ] Include path configuration
- [ ] Test script on fresh environment

## Issues Encountered
<!-- Document any issues encountered and how they were resolved -->

### Minor Bugs Fixed
- [ ] List bug 1 and fix
- [ ] List bug 2 and fix

### Major Issues (Create Separate Issues)
- [ ] Issue 1 (link to new issue)
- [ ] Issue 2 (link to new issue)

## Verification Checklist
- [ ] Code runs without errors
- [ ] Training completes successfully
- [ ] Model checkpoints are saved correctly
- [ ] Testing/evaluation works
- [ ] All paths are configurable or documented
- [ ] Dependencies are documented
- [ ] Setup process is reproducible

## Documentation Deliverables
- [ ] Document setup process
- [ ] List of dependencies with versions
- [ ] Known issues and workarounds
- [ ] Hardware requirements
- [ ] Estimated time and resource requirements
- [ ] Setup script (optional)

## Additional Notes
<!-- Add any platform-specific notes or additional considerations -->
