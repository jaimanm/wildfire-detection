---
name: Data Exploration Notebook
about: Create a notebook for exploring and analyzing the Sen2Fire dataset
title: '[DATA] '
labels: 'data-exploration, documentation'
assignees: ''
---

## Objective
Explore and analyze the Sen2Fire dataset to better understand its characteristics and inform model development decisions.

## Requirements

### Visualizations
- [ ] Create visualizations for different band combinations:
  - [ ] RGB (bands 3, 2, 1)
  - [ ] SWIR composite (bands 11, 7, 3)
  - [ ] NBR (Normalized Burn Ratio) visualization
  - [ ] NDVI (Normalized Difference Vegetation Index) visualization
  - [ ] Other band combinations as deemed interesting
- [ ] Visualize ground truth labels overlaid on imagery
- [ ] Create histograms or distributions of spectral values
- [ ] Visualize spatial patterns in the dataset

### Data Analysis
- [ ] Examine which spectral bands correlate best with ground truth fire labels
- [ ] Analyze value ranges for each spectral band
- [ ] Identify potential data quality issues (missing values, outliers, etc.)
- [ ] Compare aerosol band impact on different composites
- [ ] Calculate and visualize class balance (fire vs non-fire pixels)
- [ ] Analyze spatial characteristics (patch sizes, fire extent patterns)

### Documentation
- [ ] Include markdown cells explaining:
  - [ ] Dataset structure and Sen2Fire specifics
  - [ ] Purpose of each visualization
  - [ ] Insights gained from each analysis
  - [ ] Rationale for chosen analysis methods
  - [ ] Conclusions and recommendations for model development

### Technical Details
- [ ] Load data using the Sen2Fire_Dataset class
- [ ] Work with multiple input modes (mode parameter 0-11)
- [ ] Save the notebook in an appropriate location
- [ ] Ensure notebook runs without errors
- [ ] Include proper imports and dependencies

## Dataset Reference
- **Dataset Path**: `../Sen2Fire/` (adjust as needed)
- **List Files**: `./dataset/train.txt`, `./dataset/val.txt`, `./dataset/test.txt`
- **Input Modes**: See `dataset/Sen2Fire_Dataset.py` for available modes

## Deliverables
- [ ] Jupyter notebook with complete analysis
- [ ] Visualizations demonstrating understanding of the dataset
- [ ] Markdown documentation explaining findings
- [ ] Saved figures/plots (if applicable)

## Additional Notes
<!-- Add any specific areas of focus or questions to investigate -->
