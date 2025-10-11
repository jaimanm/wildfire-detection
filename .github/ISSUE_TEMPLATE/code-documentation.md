---
name: Add Code Comments/Documentation
about: Improve code documentation with verbose inline comments
title: '[DOCS] '
labels: 'documentation, code-quality'
assignees: ''
---

## Objective
Add comprehensive inline comments to make the codebase accessible to people unfamiliar with the model building, training, testing, and validation process.

## Files to Document

### Priority Files
- [ ] `train.py` / `train.ipynb` - Training script/notebook
  - [ ] Hyperparameter explanations
  - [ ] Training loop logic
  - [ ] Loss function and metrics
  - [ ] Data loading and batching
  - [ ] Model checkpointing
  - [ ] Validation procedure
  
- [ ] `model/Networks.py` - Model architecture
  - [ ] U-Net architecture components (DoubleConv, Down, Up, OutConv)
  - [ ] Forward pass explanation
  - [ ] Input/output shapes at each layer
  - [ ] Quantum circuit components (if uncommented)
  - [ ] Design decisions and rationale
  
- [ ] `dataset/Sen2Fire_Dataset.py` - Dataset class
  - [ ] Sen2Fire dataset structure
  - [ ] Data loading and preprocessing
  - [ ] Band combination functions
  - [ ] Normalization approach
  - [ ] Input mode explanations (0-11)
  
- [ ] `test_all.py` - Testing/evaluation script
  - [ ] Evaluation metrics
  - [ ] Inference process
  - [ ] Scene processing logic
  - [ ] Output generation

### Secondary Files
- [ ] `utils/tools.py` - Utility functions
  - [ ] Metric calculation functions
  - [ ] Data processing utilities
  - [ ] Visualization helpers
  
- [ ] `Train.py` - Alternative training script
  - Similar to train.py if significantly different

## Documentation Standards

### Comment Guidelines
- [ ] Add module-level docstrings explaining file purpose
- [ ] Add function/class docstrings with:
  - Purpose description
  - Parameter descriptions with types
  - Return value descriptions with types
  - Example usage (where helpful)
- [ ] Add inline comments for:
  - Complex logic or algorithms
  - Non-obvious design decisions
  - Hyperparameter choices
  - Data shape transformations
  - Metric calculations
- [ ] Explain domain-specific concepts:
  - Spectral bands and indices (RGB, SWIR, NBR, NDVI)
  - Satellite imagery specifics
  - Wildfire detection metrics

### Style Requirements
- [ ] Use clear, concise language
- [ ] Maintain consistency with existing comment style
- [ ] Avoid obvious comments (e.g., "increment counter")
- [ ] Focus on "why" rather than just "what"
- [ ] Keep comments up-to-date with code changes

## Example Documentation Improvements

### Before:
```python
mode = 5
```

### After:
```python
# Input mode defines the spectral band combination used for training
# Mode 5 = swir_aerosol: Uses SWIR composite (bands 11, 7, 3) plus aerosol band
# See dataset/Sen2Fire_Dataset.py for all available modes (0-11)
mode = 5
```

## Verification Checklist
- [ ] All functions have docstrings
- [ ] Complex logic blocks have explanatory comments
- [ ] Variable names are clear or commented
- [ ] Design decisions are documented
- [ ] Code is understandable to a new contributor
- [ ] No outdated or misleading comments

## Additional Notes
<!-- Add specific areas that need more documentation or clarification -->
