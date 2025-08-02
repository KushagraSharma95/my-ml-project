# RSNA Pneumonia Detection

## Problem Description and Business Context
Automatic detection of pneumonia in chest X-rays can augment radiologists by identifying suspicious findings in high-throughput or resource-constrained environments. The project tackled localization and classification of pneumonia opacities using the RSNA challenge dataset.

## Data Source and Exploration
Used annotated chest X-ray images (DICOM) with radiologist bounding boxes indicating pneumonia. Explored class distribution, box size variability, and examples of normal vs abnormal and confounding cases (e.g., fibrosis, effusions).

## Preprocessing
- Loaded DICOMs via pydicom, applied lung windowing, normalized intensities, and resized images.
- Augmentation: horizontal flips, small rotations, brightness/contrast jitter.
- Prepared bounding box annotations for detectors; implemented fallback classification baseline (pneumonia presence vs absence) to sanity-check downstream detection.

## Modeling Pipeline & Experiments
### Classification Baseline
Fine-tuned a pretrained ResNet50 for pneumonia presence, achieving strong AUC (~0.85) and highlighting learnable patterns.

### Object Detection
- **Mask R-CNN:** Used a two-stage detector with pretrained COCO weights; tuned anchors, trained on 512x512 inputs. Achieved mAP ≈ 0.23 at IoU ≥ 0.5 after fine-tuning.
- **YOLOv3:** Explored one-stage alternatives; faster inference but slightly lower mAP (~0.18). Provided insight into trade-offs between speed and localization granularity.
- **Ensembling:** Combined Mask R-CNN and YOLO outputs to capture more cases, improving recall at the cost of precision.

## Evaluation
- Detection recall for pneumonia-positive images ~80%; localization mAP moderate due to subtle and small opacities.
- Classification-level AUC using detection presence threshold reached ~0.90.
- Adjusted confidence thresholds for clinical trade-offs (favor sensitivity over specificity).

## Interpretability
- Used Grad-CAM on classification model to validate focus regions.
- Analyzed common false positives (e.g., pleural effusion, scarring) and false negatives (subtle apical opacities).

## Tools & Technologies
Python, pydicom, Keras/TensorFlow (Mask R-CNN), Darknet/YOLOv3, OpenCV, imgaug, matplotlib, Jupyter, Git.

## Conclusions and Lessons Learned
- Transfer learning was critical; starting from pretrained weights accelerated convergence.
- Balancing sensitivity and specificity is domain-dependent: higher sensitivity favored for screening.
- Ensemble detection captures diverse failure modes but requires careful thresholding.
- Explainability via bounding boxes and heatmaps builds stakeholder trust.
