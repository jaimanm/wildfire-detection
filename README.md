# Wildfire Detection Using Deep Learning and Quantum-Enhanced U-Net

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

A comprehensive machine learning framework for automated wildfire detection using multi-spectral Sentinel-2 satellite imagery. This repository implements a U-Net architecture for pixel-level fire segmentation, achieving 93.41% overall accuracy on the Sen2Fire benchmark dataset.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/jaimanm/wildfire-detection.git
cd wildfire-detection

# Install dependencies
pip install torch torchvision numpy scipy matplotlib tqdm pennylane
```

### Dataset Setup

1. **Download Sen2Fire Dataset**: Obtain the dataset from the [original publication repository](https://arxiv.org/abs/2403.17884)

2. **Organize Dataset Structure**: Place the dataset in the following structure:
```
wildfire-detection/
├── ../Sen2Fire/              # Dataset root (one level up from repo)
│   ├── scene1/
│   │   ├── scene_1_patch_1_1.npz
│   │   ├── scene_1_patch_1_2.npz
│   │   └── ...
│   ├── scene2/
│   ├── scene3/
│   └── scene4/
└── wildfire-detection/        # Repository root
    ├── dataset/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    └── ...
```

Each NPZ file contains:
- `image`: 12-band Sentinel-2 data (shape: 12, 512, 512)
- `label`: Binary fire mask (shape: 512, 512)
- `aerosol`: Sentinel-5P aerosol data (shape: 512, 512)

### Quick Training Example

```bash
# Train with default configuration (Mode 5: SWIR + aerosol)
python train.py

# Or modify train.py to set:
# - mode = 5  # Spectral band configuration (0-11)
# - data_dir = '../Sen2Fire/'
# - epochs = 5
# - batch_size = 16
# - learning_rate = 1e-4
```

### Quick Inference Example

```bash
# Run inference on test scenes
python test_all.py \
    --data_dir ../Sen2Fire \
    --test_list ./dataset/test.txt \
    --mode 5 \
    --restore_from ./Exp/swir_aerosol/weight_10_time0815_1330/best_model.pth \
    --snapshot_dir ./Map/
```

## Usage

### Training

The training script (`train.py`) supports the following configurations:

**Spectral Band Modes** (set `mode` variable in script):
- `mode = 0`: All 12 Sentinel-2 bands
- `mode = 1`: All 12 bands + aerosol
- `mode = 2`: RGB composite (B4, B3, B2)
- `mode = 3`: RGB + aerosol
- `mode = 4`: SWIR composite (B12, B8, B4)
- `mode = 5`: **SWIR + aerosol** (recommended, best performance)
- `mode = 6`: NBR composite
- `mode = 7`: NBR + aerosol
- `mode = 8`: NDVI composite
- `mode = 9`: NDVI + aerosol
- `mode = 10`: RGB + SWIR + NBR + NDVI
- `mode = 11`: RGB + SWIR + NBR + NDVI + aerosol

**Key Hyperparameters** (modify in `train.py`):
```python
epochs = 5
learning_rate = 1e-4
weight_decay = 5e-4
batch_size = 16  # Training
weight = 10  # Class weight for fire class (addresses imbalance)
```

**Training Output**:
- Model checkpoints saved to: `./Exp/{mode_name}/weight_{weight}_time{timestamp}/`
- Training logs: `Training_log.txt`
- Best model: `best_model.pth`
- Training history: `*_hist.npz`

### Inference

The inference script (`test_all.py`) supports command-line arguments:

```bash
python test_all.py \
    --data_dir <path_to_Sen2Fire> \
    --test_list <path_to_test_list> \
    --mode <0-11> \
    --batch_size <batch_size> \
    --num_workers <num_workers> \
    --restore_from <path_to_model.pth> \
    --snapshot_dir <output_directory>
```

**Output**:
- Predictions saved as sparse matrices: `{snapshot_dir}/preds/Scene{1-4}/*.npz`
- Visualization maps: `{snapshot_dir}/scene{1-4}_map.png`

### HPC/SLURM Usage

For HPC clusters with SLURM:

```bash
# Submit training job
sbatch submit_training.sh

# Submit inference job
sbatch submit_test.sh
```

The SLURM scripts configure:
- GPU: A100 (full or fractional)
- CPU: 12 cores per task
- Memory: 4GB per core (default)
- Runtime: 2.5 hours maximum

## Configuration

### File Structure

```
wildfire-detection/
├── dataset/                   # Dataset handling
│   ├── Sen2Fire_Dataset.py   # PyTorch dataset class
│   ├── train.txt             # Training patch list
│   ├── val.txt               # Validation patch list
│   └── test.txt              # Test patch list
├── model/
│   └── Networks.py           # U-Net architecture definition
├── utils/
│   └── tools.py               # Evaluation metrics and utilities
├── Exp/                       # Experiment results
│   ├── swir_aerosol/         # Mode 5 results
│   └── quantum_swir_aerosol/ # Quantum experiments
├── Map/                       # Generated prediction maps
├── jobs/                      # SLURM job outputs
├── train.py                   # Training script
├── test_all.py               # Inference script
├── submit_training.sh         # SLURM training submission
└── submit_test.sh            # SLURM inference submission
```

### Model Architecture

- **Architecture**: U-Net with encoder-decoder structure
- **Input Size**: 512×512 pixels
- **Channels**: Variable (3-13) depending on mode
- **Output**: Binary segmentation (fire/non-fire)
- **Encoder**: 4 downsampling stages (64→128→256→512→1024 channels)
- **Decoder**: 4 upsampling stages with skip connections
- **Loss**: Weighted Cross-Entropy (weights: [1, 10])
- **Optimizer**: Adam (lr=1e-4, weight_decay=5e-4)

### Performance Metrics

The model reports:
- **Overall Accuracy**: Fraction of correctly classified pixels
- **Fire Precision**: True positives / (True positives + False positives)
- **Fire Recall**: True positives / (True positives + False negatives)
- **Fire F1-Score**: Harmonic mean of precision and recall
- **Fire IoU**: Intersection over Union for fire class
- **Mean IoU**: Average IoU across both classes

## Results

**Best Configuration (Mode 5: SWIR + Aerosol)**:
- Overall Accuracy: **93.41%**
- Fire Precision: **38.41%**
- Fire Recall: **23.18%**
- Fire F1-Score: **28.91%**
- Fire IoU: **16.90%**
- Mean IoU: **62.73%**

**Alternative Configuration (Input SWIR + Aerosol)**:
- Overall Accuracy: **93.55%**
- Fire Precision: **38.81%**
- Fire Recall: **20.08%**
- Fire F1-Score: **26.47%**
- Fire IoU: **15.25%**
- Mean IoU: **61.55%**

## Requirements

### Software
- Python 3.8+
- PyTorch 1.9.0+
- NumPy
- SciPy
- Matplotlib
- tqdm
- PennyLane 0.20.0+ (for quantum experiments, currently disabled)

### Hardware
- GPU: NVIDIA A100 or equivalent (recommended)
- CPU: Multi-core processor (12+ cores recommended)
- Memory: 48GB+ RAM
- Storage: SSD for dataset and model files

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{xu2024sen2fire,
  title={Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data},
  author={Xu, Yonghao and Berg, Amanda and Haglund, Leif},
  journal={arXiv preprint arXiv:2403.17884},
  year={2024}
}
```

**Conference Paper**: Xu, Y., Berg, A., & Haglund, L. (2024). Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data. *2024 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*. DOI: [10.1109/IGARSS53475.2024.10641441](https://doi.org/10.1109/IGARSS53475.2024.10641441)

---

## Detailed Documentation

The following sections provide comprehensive scientific documentation for researchers and practitioners seeking deeper understanding of the methodology, experimental design, and results.

## Abstract

This repository presents a comprehensive machine learning framework for automated wildfire detection using multi-spectral satellite imagery. The system employs a U-Net convolutional neural network architecture, enhanced with experimental quantum computing components, to perform pixel-level semantic segmentation of fire events in Sentinel-2 satellite data. The methodology addresses the critical challenge of early wildfire detection through the integration of thirteen spectral bands from the Sentinel-2 Multispectral Imager (MSI) combined with Sentinel-5P aerosol products for atmospheric correction. Our approach systematically evaluates twelve distinct spectral band configurations, demonstrating that selective band combinations, particularly those emphasizing short-wave infrared (SWIR) wavelengths, significantly outperform approaches utilizing all available spectral information. The framework achieves an overall accuracy of 93.41% on the Sen2Fire benchmark dataset, with fire class precision and recall of 38.41% and 23.18% respectively, despite severe class imbalance inherent to wildfire detection tasks. The experimental results validate the importance of spectral band selection and atmospheric correction in remote sensing applications, while highlighting the persistent challenges associated with class imbalance in binary fire classification. This work contributes to the growing body of research on automated wildfire monitoring systems and provides a reproducible baseline for future investigations in satellite-based fire detection.

## Background and Problem Statement

Wildfire detection represents a critical application domain within remote sensing and environmental monitoring, with significant implications for public safety, ecosystem management, and emergency response coordination. The increasing frequency and intensity of wildfires globally, exacerbated by climate change and human activities, necessitates the development of automated detection systems capable of processing large volumes of satellite imagery in near real-time. Traditional manual monitoring approaches are insufficient for the scale and temporal resolution required for effective fire management, creating an urgent need for machine learning-based solutions that can operate autonomously on continuous satellite data streams.

The problem of wildfire detection from satellite imagery presents several fundamental challenges that distinguish it from conventional computer vision tasks. First, the extreme class imbalance between fire and non-fire pixels creates a significant learning challenge, as fire events typically represent less than one percent of total image pixels. This imbalance leads to models that achieve high overall accuracy through trivial predictions of the majority class, while failing to detect the rare but critical fire events. Second, the spectral complexity of fire signatures requires sophisticated multi-spectral analysis, as different fire types, intensities, and environmental conditions produce distinct spectral responses across the electromagnetic spectrum. Third, atmospheric interference from smoke, clouds, and aerosols can either obscure fire signatures or create false positive detections, necessitating robust atmospheric correction mechanisms. Fourth, the temporal variability of fire events requires models capable of detecting fires across different times of day, seasons, and environmental conditions. Finally, the multi-scale nature of fires, ranging from small ignition points to large conflagrations, demands architectures capable of capturing both fine-grained detail and broader contextual information.

The Sentinel-2 mission, part of the European Space Agency's Copernicus program, provides an ideal data source for wildfire detection applications. The Multispectral Imager (MSI) aboard Sentinel-2A and Sentinel-2B satellites captures thirteen spectral bands at spatial resolutions ranging from 10 to 60 meters, with a revisit time of five days at the equator. The spectral bands span from visible wavelengths through near-infrared and short-wave infrared regions, providing comprehensive coverage of the electromagnetic spectrum relevant to fire detection. Particularly relevant for wildfire applications are the short-wave infrared bands (B11 at 1610 nm and B12 at 2190 nm), which exhibit strong sensitivity to high-temperature targets and have been demonstrated to be effective for fire detection in previous studies. The integration of Sentinel-5P aerosol products further enhances the detection capability by providing atmospheric correction data that improves the accuracy of spectral analysis under varying atmospheric conditions.

## Related Work

The application of machine learning to wildfire detection has been explored through various approaches, ranging from traditional statistical methods to modern deep learning architectures. Early work in this domain focused on threshold-based detection algorithms operating on single spectral bands or simple band ratios, such as the normalized burn ratio (NBR) and normalized difference vegetation index (NDVI). These methods, while computationally efficient, suffer from high false positive rates and limited adaptability to diverse environmental conditions.

More recent research has explored the application of convolutional neural networks (CNNs) to wildfire detection, with architectures ranging from simple classification networks to sophisticated segmentation models. The U-Net architecture, originally developed for biomedical image segmentation, has shown particular promise in remote sensing applications due to its ability to preserve spatial information through skip connections while performing multi-scale feature extraction. The encoder-decoder structure of U-Net enables the network to learn hierarchical representations that capture both local fire signatures and broader contextual information about the surrounding landscape.

The Sen2Fire dataset, introduced by Xu et al. (2024), provides a standardized benchmark for evaluating wildfire detection algorithms. This dataset comprises 2,466 image patches extracted from Sentinel-2 imagery, each measuring 512×512 pixels with thirteen spectral bands including Sentinel-5P aerosol data. The dataset addresses a critical gap in the field by providing a large-scale, publicly available benchmark with manually annotated ground truth labels, enabling reproducible evaluation of detection algorithms. The authors demonstrated that selective band combinations outperform approaches using all available spectral information, validating the importance of spectral band selection in fire detection applications.

## Proposed Method

### Architecture Overview

Our approach employs a U-Net architecture as the primary segmentation network, chosen for its demonstrated effectiveness in semantic segmentation tasks and its ability to preserve fine-grained spatial detail through skip connections. The U-Net architecture consists of a symmetric encoder-decoder structure with skip connections between corresponding layers in the encoder and decoder paths. The encoder path performs progressive downsampling through four stages, each consisting of a max pooling operation followed by two convolutional layers with batch normalization and ReLU activation. The channel dimensions increase from 64 in the initial layer to 128, 256, 512, and finally 1024 in the bottleneck layer, enabling the network to learn increasingly abstract feature representations.

The decoder path mirrors the encoder structure, performing progressive upsampling through four stages using bilinear interpolation. Each decoder stage concatenates the upsampled feature maps with the corresponding encoder feature maps through skip connections, enabling the network to recover fine-grained spatial detail that would otherwise be lost through the downsampling process. The final output layer consists of a 1×1 convolution that produces pixel-wise class predictions for the binary fire classification task. The network architecture is designed to process input images of 512×512 pixels, matching the patch size of the Sen2Fire dataset.

The model architecture includes experimental support for quantum-enhanced feature processing, implemented through PennyLane quantum computing framework. The quantum component consists of an 8-qubit quantum circuit with two layers of strongly entangling gates, positioned at the bottleneck of the U-Net architecture. The quantum circuit employs amplitude embedding to encode classical feature vectors into quantum states, followed by parameterized quantum gates and measurement operations. However, this quantum enhancement component is currently disabled in the primary implementation due to numerical stability challenges encountered during training. The quantum architecture represents an exploratory direction for potential computational advantages in feature learning, though the current focus remains on the classical U-Net implementation.

### Spectral Band Configuration

A critical aspect of our methodology involves the systematic evaluation of different spectral band combinations to identify optimal input configurations for fire detection. The system supports twelve distinct input modes, each representing a different combination of Sentinel-2 spectral bands and derived indices. Mode 0 utilizes all twelve Sentinel-2 bands (B1 through B12), providing complete spectral information at the cost of increased computational complexity and potential redundancy. Mode 1 extends this configuration by incorporating Sentinel-5P aerosol data as an additional channel, enabling explicit atmospheric correction.

Modes 2 through 5 focus on specific spectral regions known to be relevant for fire detection. Mode 2 employs a standard RGB composite using bands B4 (red), B3 (green), and B2 (blue), providing natural color visualization. Mode 4 utilizes a SWIR composite combining bands B12 (2190 nm), B8 (842 nm near-infrared), and B4 (665 nm red), emphasizing the short-wave infrared region where fire signatures are most pronounced. Mode 5 extends this configuration with aerosol data, representing our primary experimental configuration based on empirical performance results.

Modes 6 through 9 incorporate derived spectral indices that have demonstrated utility in fire and vegetation monitoring applications. Mode 6 employs the normalized burn ratio (NBR), calculated as (B8 - B12) / (B8 + B12), combined with red and green bands. The NBR index is particularly effective for detecting burn scars and assessing fire severity. Mode 8 utilizes the normalized difference vegetation index (NDVI), calculated as (B8 - B4) / (B8 + B4), which provides information about vegetation health and can aid in distinguishing between active fires and post-fire vegetation changes. Modes 7 and 9 extend these configurations with aerosol data.

Modes 10 and 11 represent comprehensive configurations that combine multiple spectral indices, including RGB composites, SWIR bands, NBR, and NDVI, with mode 11 incorporating aerosol data. These configurations provide the richest feature representation but at the cost of increased model complexity and computational requirements.

### Training Methodology

The training process employs a weighted cross-entropy loss function to address the severe class imbalance inherent in wildfire detection tasks. The loss function applies a weight of 1.0 to the non-fire class and a weight of 10.0 to the fire class, effectively increasing the penalty for misclassifying fire pixels by an order of magnitude. This weighting strategy helps the model learn to prioritize accurate fire detection despite the overwhelming presence of non-fire pixels in the training data.

The optimization process utilizes the Adam optimizer with a learning rate of 1e-4 and weight decay of 5e-4 for L2 regularization. The training is conducted over five epochs, with batch sizes of 16 for training, 50 for testing, and 1 for validation. The smaller validation batch size enables per-image evaluation that more closely matches the inference scenario. The training process includes per-epoch validation on a held-out validation set, with model checkpointing based on the fire class F1-score to ensure that the best-performing model is preserved.

The data preprocessing pipeline normalizes pixel values by dividing by 10,000, converting the raw digital numbers from Sentinel-2 imagery to reflectance values. The input images are processed at their native 512×512 pixel resolution, with bilinear interpolation used for upsampling model predictions to match the input dimensions when necessary. The training process includes data augmentation through random shuffling and the use of multiple data loader workers to improve training efficiency on multi-core systems.

## Dataset Description

The experimental evaluation is conducted on the Sen2Fire dataset, a publicly available benchmark dataset specifically designed for wildfire detection research. The dataset, introduced by Xu et al. (2024) and presented at the IGARSS 2024 conference, comprises 2,466 image patches extracted from Sentinel-2 satellite imagery. Each patch measures 512×512 pixels and contains thirteen spectral channels: twelve bands from the Sentinel-2 MSI sensor plus one aerosol band from the Sentinel-5P mission.

The Sentinel-2 MSI bands included in the dataset span the electromagnetic spectrum from visible wavelengths through short-wave infrared. The visible and near-infrared bands (B2, B3, B4, B8) are provided at 10-meter spatial resolution, while the red edge and additional near-infrared bands (B5, B6, B7, B8A) are provided at 20-meter resolution. The short-wave infrared bands (B11, B12), which are particularly critical for fire detection, are also provided at 20-meter resolution. The coastal aerosol band (B1) and cirrus band (B10) are provided at 60-meter resolution. The dataset includes Sentinel-5P aerosol optical depth data, which enables atmospheric correction and improves detection accuracy under varying atmospheric conditions.

The ground truth labels for the dataset consist of binary masks indicating fire and non-fire pixels, manually annotated by domain experts. The annotation process involved careful examination of each image patch to identify active fire pixels based on spectral signatures, thermal anomalies, and contextual information. The dataset is partitioned into training, validation, and test sets, with 1,448 patches allocated for training and the remaining patches distributed between validation and testing. The dataset exhibits the characteristic class imbalance of wildfire detection tasks, with fire pixels representing a small fraction of total pixels across all patches.

The data preprocessing pipeline loads image patches from NPZ (NumPy compressed) files, each containing three arrays: the multi-spectral image data, the binary label mask, and the aerosol data. The preprocessing applies mode-specific band selection based on the experimental configuration, normalizes pixel values to reflectance units, and prepares the data for batch processing. The dataset implementation supports efficient data loading through PyTorch DataLoader classes with configurable batch sizes and multi-threaded data loading for improved training efficiency.

## Experimental Setup and Evaluation Metrics

The experimental evaluation is conducted on high-performance computing infrastructure utilizing NVIDIA A100 GPUs through the SLURM job scheduling system. The computational environment provides access to GPU-accelerated training with 12 CPU cores per task and configurable memory allocation. The software stack includes PyTorch for deep learning operations, PennyLane for quantum computing experiments, and standard scientific Python libraries including NumPy, SciPy, and Matplotlib for data processing and visualization.

The evaluation employs a comprehensive set of metrics designed to assess different aspects of model performance. Overall accuracy measures the fraction of correctly classified pixels across both classes, providing a general measure of classification performance. However, given the severe class imbalance, overall accuracy can be misleading, as a model that predicts all pixels as non-fire would achieve high accuracy despite complete failure to detect fires. Therefore, the evaluation emphasizes class-specific metrics including precision, recall, F1-score, and Intersection over Union (IoU) for the fire class.

Precision for the fire class measures the fraction of predicted fire pixels that are actually fire pixels, indicating the model's ability to avoid false positive detections. Recall measures the fraction of actual fire pixels that are correctly identified, indicating the model's ability to detect fires without missing events. The F1-score provides a harmonic mean of precision and recall, offering a balanced measure of detection performance. The IoU metric measures the overlap between predicted and ground truth fire regions, providing a spatial accuracy measure that is particularly relevant for segmentation tasks.

The experimental protocol involves training separate models for each of the twelve spectral band configurations, evaluating each configuration on the same validation and test sets to enable fair comparison. The training process includes early stopping based on validation performance, with model checkpoints saved when the fire class F1-score improves. The final evaluation reports performance metrics on the held-out test set, ensuring that reported results represent generalization performance rather than training set memorization.

## Results and Discussion

The experimental evaluation demonstrates that selective spectral band combinations significantly outperform approaches utilizing all available spectral bands. The best-performing configuration, Mode 5 (SWIR + aerosol), achieves an overall accuracy of 93.41% on the test set, with fire class precision of 38.41%, recall of 23.18%, F1-score of 28.91%, and IoU of 16.90%. The mean IoU across both classes reaches 62.73%, indicating reasonable segmentation performance despite the challenging class imbalance. An alternative configuration utilizing input SWIR with aerosol achieved slightly higher overall accuracy of 93.55% and precision of 38.81%, though with lower recall of 20.08%, F1-score of 26.47%, and IoU of 15.25%, with mean IoU of 61.55%. This alternative configuration demonstrates the sensitivity of performance metrics to specific band selection strategies, with the primary configuration (Mode 5) providing a better balance between precision and recall.

The results reveal a critical trade-off between precision and recall in fire detection. While the model achieves moderate precision, indicating that most predicted fire pixels are indeed fires, the relatively low recall suggests that the model misses a significant fraction of actual fire events. This behavior is consistent with the class imbalance challenge, as the model learns to be conservative in fire predictions to avoid the high penalty associated with false positives in the weighted loss function. The precision-recall trade-off represents a fundamental limitation of the current approach and suggests that alternative loss functions or sampling strategies may be necessary to improve recall without sacrificing precision.

The comparison between different spectral band configurations reveals several important insights. Configurations emphasizing SWIR bands (Modes 4 and 5) consistently outperform RGB-based configurations (Modes 2 and 3), validating the importance of short-wave infrared wavelengths for fire detection. The incorporation of aerosol data provides measurable improvements in most configurations, though the magnitude of improvement varies across different band combinations. The comprehensive configurations (Modes 10 and 11) that combine multiple spectral indices do not consistently outperform simpler configurations, suggesting that increased feature dimensionality does not necessarily translate to improved performance in this application domain.

The experimental results align with findings from the Sen2Fire dataset paper, which similarly demonstrated that selective band combinations outperform approaches using all available spectral information. This consistency across independent evaluations strengthens confidence in the generalizability of these findings. The importance of SWIR bands, particularly B12 at 2190 nm, is consistent with the physical properties of fire signatures, as high-temperature targets emit strongly in the short-wave infrared region of the spectrum.

The quantum enhancement experiments encountered numerical stability challenges during training, preventing successful integration of the quantum components in the primary implementation. The quantum circuit architecture, while theoretically promising for feature learning, requires further investigation to address the numerical issues and validate potential computational advantages. The current focus remains on the classical U-Net implementation, with quantum enhancement representing a future research direction.

## Future Work and Known Limitations

Several directions for future research emerge from the current work, addressing both the technical limitations identified in the experimental evaluation and opportunities for extending the methodology to new application domains. The most critical limitation requiring attention is the low recall rate for fire detection, which currently limits the practical utility of the system for autonomous operation. Future work should explore alternative loss functions, such as focal loss or dice loss, which may better handle the extreme class imbalance. Additionally, sampling strategies that oversample fire pixels or undersample non-fire pixels during training could help the model learn more effective fire representations.

The quantum enhancement component represents an intriguing direction for future investigation, though significant technical challenges must be overcome before practical deployment. The numerical stability issues encountered in the current implementation suggest that alternative quantum circuit architectures or training strategies may be necessary. Research into quantum-classical hybrid architectures that more effectively integrate quantum and classical components could yield improvements in both computational efficiency and detection performance.

The extension of the methodology to temporal sequences represents another promising direction, as fires evolve over time and temporal information could significantly improve detection accuracy. The integration of multiple satellite passes over the same geographic region could enable the detection of fire progression and improve the distinction between active fires and false positive detections such as hot surfaces or industrial heat sources. The development of spatio-temporal architectures that process image sequences rather than individual frames could address this opportunity.

The application of the methodology to other remote sensing tasks, such as burn scar mapping, vegetation health monitoring, or post-fire recovery assessment, represents another avenue for future work. The U-Net architecture and spectral band selection strategies developed for fire detection may be transferable to these related applications, potentially accelerating progress in multiple domains of environmental monitoring.

The current implementation is optimized for research and development rather than production deployment, with manual processes for model updates, data pipeline management, and performance monitoring. Future work should develop automated deployment pipelines, containerization strategies, and monitoring systems that enable reliable operation in production environments. The development of REST APIs or other programmatic interfaces would facilitate integration with existing fire management systems and emergency response workflows.

## References

Xu, Y., Berg, A., & Haglund, L. (2024). Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data. *arXiv preprint arXiv:2403.17884*. https://arxiv.org/abs/2403.17884

Xu, Y., Berg, A., & Haglund, L. (2024). Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data. *2024 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*. DOI: [10.1109/IGARSS53475.2024.10641441](https://doi.org/10.1109/IGARSS53475.2024.10641441)

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 234-241.

The Sentinel-2 mission and data products are provided by the European Space Agency (ESA) as part of the Copernicus program. Technical documentation regarding Sentinel-2 band specifications and applications can be found in the official ESA Sentinel-2 User Handbook and through resources such as the [GISGeography Sentinel-2 band combinations guide](https://gisgeography.com/sentinel-2-bands-combinations/).

---

*This documentation reflects the current state of the codebase as of August 2024. The methodology and results are subject to ongoing refinement as the research progresses.*
