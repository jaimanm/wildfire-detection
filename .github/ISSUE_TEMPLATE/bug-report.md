---
name: Bug Report
about: Report a bug or unexpected behavior in the codebase
title: '[BUG] '
labels: 'bug'
assignees: ''
---

## Bug Description
<!-- A clear and concise description of what the bug is -->

## Expected Behavior
<!-- What you expected to happen -->

## Actual Behavior
<!-- What actually happened -->

## Steps to Reproduce
1. 
2. 
3. 

## Error Messages
<!-- If applicable, paste the full error message and stack trace -->
```
Paste error message here
```

## Environment
- **OS**: <!-- e.g., Ubuntu 20.04, Windows 10, macOS 12 -->
- **Python Version**: 
- **PyTorch Version**: 
- **CUDA Version** (if applicable): 
- **Hardware** (CPU/GPU): 

## Code Context
<!-- Which file(s) and function(s) are involved? -->
- **File(s)**: 
- **Function/Class**: 
- **Line Number(s)**: 

## Affected Components
<!-- Mark all that apply -->
- [ ] Training (`train.py` / `train.ipynb`)
- [ ] Testing/Evaluation (`test_all.py`)
- [ ] Model Architecture (`model/Networks.py`)
- [ ] Dataset Loading (`dataset/Sen2Fire_Dataset.py`)
- [ ] Utilities (`utils/tools.py`)
- [ ] Submit Scripts
- [ ] Other: 

## Input Mode (if applicable)
<!-- If the bug is mode-specific, specify which mode -->
- [ ] Mode 0: all_bands
- [ ] Mode 1: all_bands_aerosol
- [ ] Mode 2: rgb
- [ ] Mode 3: rgb_aerosol
- [ ] Mode 4: swir
- [ ] Mode 5: swir_aerosol
- [ ] Mode 6: nbr
- [ ] Mode 7: nbr_aerosol
- [ ] Mode 8: ndvi
- [ ] Mode 9: ndvi_aerosol
- [ ] Mode 10: rgb_swir_nbr_ndvi
- [ ] Mode 11: rgb_swir_nbr_ndvi_aerosol
- [ ] All modes / Not mode-specific

## Minimal Reproducible Example
<!-- If possible, provide a minimal code snippet that reproduces the issue -->
```python
# Your code here
```

## Additional Context
<!-- Any additional information, screenshots, or context about the problem -->

## Possible Solution
<!-- If you have suggestions on how to fix the bug, describe them here -->

## Severity
- [ ] Critical (blocks all work)
- [ ] High (major functionality affected)
- [ ] Medium (workaround available)
- [ ] Low (minor issue)
