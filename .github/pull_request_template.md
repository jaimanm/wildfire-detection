## Description
<!-- Provide a clear and concise description of your changes -->

## Related Issue
<!-- Link to the issue this PR addresses. Use "Closes #123" or "Fixes #123" to automatically close the issue when merged -->
Closes #

## Type of Change
<!-- Mark the relevant option with an "x" -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Documentation update
- [ ] Code quality improvement (refactoring, comments, formatting)
- [ ] Data exploration/analysis
- [ ] Model improvement or new architecture
- [ ] Dataset preprocessing or augmentation
- [ ] Testing/validation

## Changes Made
<!-- List the specific changes made in this PR -->
- 
- 
- 

## Testing Performed
<!-- Describe the testing you performed to verify your changes -->
- [ ] Tested locally on CPU/GPU
- [ ] Verified model training runs without errors
- [ ] Checked model inference/testing functionality
- [ ] Validated data loading and preprocessing
- [ ] Ran existing tests (if applicable)
- [ ] Added new tests for new functionality

### Test Environment
- **Python Version**: 
- **PyTorch Version**: 
- **CUDA Version** (if applicable): 
- **Hardware** (CPU/GPU model): 

## Dataset & Model Impact
<!-- Describe how this affects the dataset handling or model -->
- [ ] Changes affect dataset loading (Sen2Fire_Dataset.py)
- [ ] Changes affect model architecture (Networks.py)
- [ ] Changes affect training procedure (train.py/train.ipynb)
- [ ] Changes affect testing/evaluation (test_all.py)
- [ ] Changes affect band combinations or input modes
- [ ] No dataset or model changes

### Band Mode Affected
<!-- If applicable, specify which input mode(s) are affected -->
- [ ] all_bands (mode 0)
- [ ] all_bands_aerosol (mode 1)
- [ ] rgb (mode 2)
- [ ] rgb_aerosol (mode 3)
- [ ] swir (mode 4)
- [ ] swir_aerosol (mode 5)
- [ ] nbr (mode 6)
- [ ] nbr_aerosol (mode 7)
- [ ] ndvi (mode 8)
- [ ] ndvi_aerosol (mode 9)
- [ ] rgb_swir_nbr_ndvi (mode 10)
- [ ] rgb_swir_nbr_ndvi_aerosol (mode 11)
- [ ] N/A or all modes

## Code Quality Checklist
<!-- Ensure your code meets quality standards -->
- [ ] Code follows existing style and conventions
- [ ] Added meaningful comments where necessary
- [ ] Updated docstrings for modified functions/classes
- [ ] No unnecessary print statements or debug code
- [ ] Removed unused imports
- [ ] Code is properly formatted
- [ ] No hardcoded paths (or documented why they exist)

## Documentation
<!-- Have you updated relevant documentation? -->
- [ ] Updated README (if applicable)
- [ ] Added/updated inline comments
- [ ] Added/updated function docstrings
- [ ] Updated configuration examples
- [ ] Added usage examples for new features
- [ ] N/A - no documentation changes needed

## Additional Notes
<!-- Any additional information that reviewers should know -->

## Screenshots (if applicable)
<!-- Add screenshots of visualizations, plots, or UI changes -->

## Checklist Before Requesting Review
- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have tested my changes thoroughly
- [ ] This PR is linked to an issue
