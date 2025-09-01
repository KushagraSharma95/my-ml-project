

# RSNA Pneumonia Detection

## Abstract

The project automates pneumonia detection in chest X-rays by identifying and localizing lung opacities using deep learning techniques. The goal is to provide an early, accurate tool for clinical decision support in healthcare imaging.

## Introduction

RSNA Pneumonia Detection Problem Description and Business Context Pneumonia is a serious lung infection and one of the leading causes of death worldwide. Early and accurate detection via chest X-rays can be life-saving.

In this project, inspired by the RSNA (Radiological Society of North America) Pneumonia Detection Challenge, we built a model to automatically detect pneumonia in chest radiographs. Specifically, the task is to identify lung opacities on chest X-ray images that are indicative of pneumonia and localize them with bounding boxes.

The business context is healthcare: a successful model could assist radiologists by flagging suspected pneumonia cases, especially in settings with high patient volumes or limited radiologist access. It could also be deployed in screening, such as in emergency rooms or during pandemics (e.g., distinguishing pneumonia in COVID-19).

The challenge, however, is significant. Pneumonia opacities can be subtle and easily confused with other lung opacities (from TB, lung cancer, edema, etc.).

Therefore, the model must be highly sensitive (not to miss pneumonia) yet specific (not to over-call every opacity as pneumonia). Data Source and Exploration We used the dataset provided by the RSNA Pneumonia Detection Challenge (2018), which was a collaboration between RSNA, the Society of Thoracic Radiology, and NIH.

This dataset was derived from the NIH ChestX-ray public dataset, with radiologist-annotated labels for pneumonia. It includes: Chest X-ray images: ~26,684 frontal chest radiographs (in DICOM format), either with evidence of pneumonia or normal.

Annotations: For each image with pneumonia, a set of one or more bounding boxes delineating the pneumonia opacities (regions of lung showing infection). Each image is labeled as either Normal, No pneumonia (but some other abnormality), or Pneumonia with bounding box(es).

Split: The challenge had a training set (with labels and boxes) and a test set (for which we had to predict and submit results). The training set included about 15,000 images with no pneumonia and 10,000 with pneumonia (some with multiple boxes).

There was also a small subset with "not normal but not pneumonia" which we treated as negative for pneumonia in training. Data came as high-resolution 16-bit grayscale images (approximately 1024 x 1024 pixels).

We first explored the metadata:

## Data

The business context is healthcare: a successful model could assist radiologists by flagging suspected pneumonia cases, especially in settings with high patient volumes or limited radiologist access. It could also be deployed in screening, such as in emergency rooms or during pandemics (e.g., distinguishing pneumonia in COVID-19).

The challenge, however, is significant. Pneumonia opacities can be subtle and easily confused with other lung opacities (from TB, lung cancer, edema, etc.).

Therefore, the model must be highly sensitive (not to miss pneumonia) yet specific (not to over-call every opacity as pneumonia). Data Source and Exploration We used the dataset provided by the RSNA Pneumonia Detection Challenge (2018), which was a collaboration between RSNA, the Society of Thoracic Radiology, and NIH.

This dataset was derived from the NIH ChestX-ray public dataset, with radiologist-annotated labels for pneumonia. It includes: Chest X-ray images: ~26,684 frontal chest radiographs (in DICOM format), either with evidence of pneumonia or normal.

Annotations: For each image with pneumonia, a set of one or more bounding boxes delineating the pneumonia opacities (regions of lung showing infection). Each image is labeled as either Normal, No pneumonia (but some other abnormality), or Pneumonia with bounding box(es).

Split: The challenge had a training set (with labels and boxes) and a test set (for which we had to predict and submit results). The training set included about 15,000 images with no pneumonia and 10,000 with pneumonia (some with multiple boxes).

There was also a small subset with "not normal but not pneumonia" which we treated as negative for pneumonia in training. Data came as high-resolution 16-bit grayscale images (approximately 1024 x 1024 pixels).

We first explored the metadata: The patient demographics were varied (the NIH dataset includes adults of all ages). There was no straightforward patient ID in the challenge data to track duplicates, but presumably each image is a unique exam.

We checked the class distribution: roughly 1/3 were pneumonia-positive. So the classes were imbalanced but not severely (in medical context, that's actually a high prevalence).

We also looked at the distribution of bounding boxes: many pneumonia cases had a single opacity, but some had two or more (indicating multi-lobar pneumonia). The box sizes varied widely – some opacities cover an entire lung lobe, others are small patches.

For visualization, we displayed some example images: Normal CXR: clear lungs (mostly dark areas), only white lines of ribs and heart border. Pneumonia CXR: presence of cloudy white opacity in lung fields.

We noted that pneumonia opacities often appear in lung peripheries or lower lobes as localized consolidations. Non-pneumonia opacities: The tricky part was images labeled "No Pneumonia" but still abnormal (e.g., scarring, fluid overload).

These have opacities that the model could confuse for pneumonia. We visualized a few – for instance, images with lung fibrosis or edema can have diffuse haziness.

Recognizing these as non-infectious is challenging even for humans. The challenge website provided a starter kit, including sample DICOM reading code and information that each image may contain 0 (normal) or multiple pneumonia bounding boxes.

We confirmed that in the labels CSV: each image ID can have multiple rows if multiple boxes (with coordinates and a label 1 for pneumonia). Normal images have a single row with a target label 0 and no box coordinates.

Preprocessing and Data Preparation blog.goodaudience.com blog.goodaudience.com Working with medical images required special preprocessing steps: DICOM to array: We used the pydicom library to load each DICOM file into a numpy array (pixel intensities). We then normalized pixel values.

DICOMs have various intensity scales; we applied a windowing for lung tissue (center ~−600 HU, width ~1500 HU) to ensure good contrast of the lung fields, then rescaled to 0-255 range for consistency. Resizing: Due to memory and model input size constraints, we resized images from ~1024x1024 down to 256x256 or 512x512 (depending on model) while preserving aspect ratio.

We also saved the scaling factor to adjust the bounding boxes accordingly (boxes' coordinates scaled down). Augmentation: To bolster the dataset, we applied augmentations focusing on realistic transformations: Horizontal flips (lungs are symmetric, though heart orientation changes, but we considered it fine).

Small rotations (±5 degrees) and translations – X-rays can be rotated slightly due to patient positioning. Variations in brightness/contrast to mimic different imaging conditions or patient sizes.

We avoided vertical flips (not realistic for human anatomy) and large rotations. Augmentations were applied on the fly during training.

Train/Validation Split: We held out ~10% of the training images as a validation set for model tuning. We stratified this split to maintain the pneumonia percentage.

Also, to avoid data leakage, if there were multiple images from one patient (not sure due to anonymization), we would ideally keep them in one set. Since we couldn't confirm patient IDs, we assumed each image independent for splitting.

Bounding Box Handling: For the object detection task, we needed to prepare data in a suitable format: For one-stage detectors (like YOLO, RetinaNet), we needed to assign each bounding box a class label (here, all boxes are "pneumonia"). We created annotation files listing each image and the normalized coordinates of its boxes.

We used Matterport's Mask R-CNN framework (built on Keras) in one approach, which required preparing a generator yielding images and corresponding masks/boxes. We wrote a custom data generator that yields an image and an array of box coordinates + class IDs for each instance.

We also tried a simpler approach where we converted the detection into a classification problem (presence/absence of pneumonia) to get an initial model, ignoring localization. For this, we labeled images as 1 if pneumonia, 0 if not, and trained a classifier on whole images.

This was mainly to have a baseline if detection proved too hard initially. We also performed data cleaning: Removed a handful of images that were unreadable or had corrupted pixels (there were very few).

Ensured that images labeled pneumonia actually had boxes in the annotation (and vice versa). The RSNA detailed info CSV indicated some images were labeled "No Lung Opacity / Not Normal" – those have label 0 (no pneumonia) but are not normal.

We included them in negative class. Modeling Pipeline and Trials We formulated this primarily as an object detection problem.

The modeling pipeline progressed in stages: 1. Baseline Classification Model: Before tackling localization, we built a simple CNN to classify an X-ray as pneumonia vs no pneumonia (no bounding boxes, just image-level label).

We used a pretrained ResNet50 (Imagenet weights) and fine- tuned it on our data (adding a global average pooling and dense layer for binary classification). This baseline achieved about 0.85 AUC on validation in distinguishing pneumonia presence.

It showed the images do have learnable patterns. However, this model doesn't indicate where pneumonia is, which is crucial for radiologist trust.

So we moved to detection. 2.

Two-Stage Detector – Mask R-CNN: We used the open-source Mask R-CNN implementation by Matterport. Mask R-CNN is an extension of Faster R- CNN that can output segmentation masks, but we only used its bounding box capability (and class output).

Key steps: Backbone: ResNet50 or ResNet101 with FPN (Feature Pyramid Network) – this extracts features at multiple scales. Region Proposal Network (RPN): generates candidate boxes.

RoI Align and Head: yields class (pneumonia vs background) and refined boxes (and masks, which we ignored). We configured it for one class ("opacity") plus background.

The anchor sizes were tuned to the scale of pneumonia we expected (roughly 10% to 50% of image height). Training: We trained Mask R-CNN on our images (512x512 input) with augmentation.

We started from COCO-pretrained weights (to leverage generic object features) and then trained for ~10 epochs on our data. The loss combined RPN loss and detection head loss.

We monitored the validation mean Average Precision (mAP) at IoU=0.5. Results: Mask R-CNN proved quite effective.

After training, it achieved a validation mAP≈0.20 @ IoU 0.5 (meaning it correctly localized ~20% of pneumonia with decent box overlap). That might sound low, but in the challenge context it was a reasonable starting point – detecting medical objects is hard.

Qualitatively, Mask R-CNN would often identify the correct lung region for large, obvious pneumonias, but missed very subtle or small ones, and occasionally proposed false positives on things like upper lung markings. We further fine-tuned it and got mAP up to ~0.25 on validation by adjusting anchors and training longer with lower learning rate.

3. One-Stage Detector – YOLOv3: To compare, we tried a one-stage approach with YOLO (You Only Look Once), following a blog that attempted YOLO for this challenge.

We used the Darknet framework through the Darkflow interface. YOLO is faster but sometimes less accurate than two-stage methods in such tasks.

We annotated images in YOLO text format (image, box coords in relative terms). After training YOLOv3 for several thousand iterations: It was faster in inference, but performance was slightly lower than Mask R- CNN.

The YOLO model had an mAP around 0.18 on validation. It often would predict only one box per image (its design tends to pick the most confident region), so multi-foci pneumonia cases were partially detected.

Given our focus on portfolio documentation, we primarily advanced the Mask R-CNN solution, but YOLO taught us about trade-offs and was a good experiment. 4.

Model Ensembling: In the Kaggle challenge context, ensembles of different detectors were used by top teams. We experimented by merging predictions of Mask R-CNN and YOLO (and our classification model as a sanity check for presence).

We used a simple approach: if either detector predicted a box with high confidence, include it. This improved detection of some cases (less missed pneumonia) but also increased false positives.

We didn't have time to fully optimize an ensemble thresholding strategy, so our final approach remained a single model. blog.goodaudience.com Training Considerations: Loss functions: For detection, the main metrics are mAP, but we monitored classification accuracy of the detector as well (did it correctly identify pneumonia images).

We saw initial overfitting (validation loss started increasing), so we applied regularization: weight decay and reduced the number of layers we fine-tuned at once (first training the network heads, then fine-tuning deeper). Class Imbalance: There's imbalance in that many images have no pneumonia.

In detection training, this shows up as many negative anchors vs relatively fewer positive anchors. Both Mask R-CNN and YOLO have mechanisms (like selecting only a subset of negative anchors for loss).

We still manually ensured that the RPN didn't get overwhelmed by negatives by adjusting anchor sampling ratios. Validation: We used the held-out set to do early stopping and hyperparameter tuning (like what confidence threshold maximized F1).

We also manually reviewed some validation outputs: In many cases, the model drew a box in the correct lung region of an opacity, which was encouraging. Some failure modes observed: apical scarring on lungs was sometimes incorrectly boxed as pneumonia, and pleural effusions (fluid at lung base) could confuse the model since both appear white.

These are also challenging for radiologists, so not surprising. Model Performance and Evaluation We evaluate performance on two levels: detection quality and classification (finding pneumonia vs not).

Detection Performance (Bounding Boxes): The standard metric is mean Average Precision (mAP) at a certain IoU threshold. We evaluated at IoU ≥ 0.5 (the box must overlap at least 50% with a true box to count as correct, which is a common criteria): Our Mask R-CNN model achieved mAP ~0.23 on the validation set at IoU 0.5.

This means on average it detected about 23% of the pneumonia instances with sufficiently overlapping boxes. This might seem low in absolute terms but was within range in the Kaggle competition (for reference, winning solutions were in the 0.3-0.4 range mAP on the private test).

We also looked at the sensitivity at image-level: i.e., how many pneumonia cases were detected at all (with at least one box). This was higher: about 0.8 recall – our model found roughly 80% of pneumonia-positive images, albeit sometimes the box wasn't tight enough or missed a second opacity (hence not full precision by detection metrics).

False positives: On average, the model would propose 1–2 boxes per normal image at a 0.5 confidence threshold. We tuned the confidence threshold to 0.5 to balance precision/recall.

At that threshold, the precision (percentage of predicted boxes that were true pneumonia) was ~0.25. If we raise threshold to be more strict (e.g., 0.7), precision rose to ~0.4 but recall dropped significantly.

In a clinical setting, missing pneumonia (false negative) is more concerning than a false alarm, so we leaned towards more sensitive settings. We visualized a few detection outputs: True positive example: an image with right lower lobe pneumonia – the model drew a box nicely around the consolidation.

False negative: a subtle pneumonia at lung apex was missed entirely. False positive: the model drew a box on the heart shadow in a normal image (likely mistaking a portion of it as opacity).

These inspections helped identify if errors were due to annotation issues (some "false positives" might be actually unannotated issues) or real mistakes. Classification Performance (image-level): We also evaluate how well the model decides if an image has pneumonia or not (ignoring localization): Using our detector's outputs, we can classify an image as pneumonia-positive if it predicts any box above a certain confidence.

At a threshold that gave us ~80% sensitivity, the image-level AUC was about 0.90 – meaning as a pneumonia vs normal classifier, it's quite good. On the challenge's internal metric (which was a combination of detection and classification), our model was respectable but short of the best.

For comparison, our initial ResNet classifier had ~0.85 AUC. The detector effectively leveraged localization to slightly improve distinguishing power.

Speed and Deployment considerations: Our final model (Mask R-CNN) ran at ~2 seconds per image on a GPU. This is fast enough for a clinical tool (screening hundreds of X-rays in minutes).

Memory usage was moderate (~2GB VRAM for 512px images in a batch of 2). We optimized the prediction code to scale back up the predicted boxes to original image size for display to radiologists.

Visualizations and Examples We included some visual results to illustrate the model's workings: Sample Detection Output: An example chest X-ray with the model's predicted bounding box drawn in red on a pneumonia region (with label "pneumonia" and confidence). This communicates how the model localizes the finding, which is crucial in medical AI for explainability.

Confusion Matrix (image-level): We plotted a confusion matrix for classification: out of (say) 200 validation images (80 pneumonia, 120 normal), the model correctly flagged most pneumonia (few false negatives) but did have some false positives (normals flagged as pneumonia). Radiologists usually prefer some false positives over false negatives, and indeed our model's operating point can be adjusted based on clinical need.

Grad-CAM: For our initial classification CNN, we generated Grad-CAM heatmaps to see where the network was "looking". Interestingly, it often highlighted the pneumonia region, which gave confidence that the model wasn't focusing on irrelevant areas.

For example, on a pneumonia image, the heatmap was bright over the opacity; on a normal image, no strong activation in lungs. Tools and Libraries Libraries: We used pydicom for image reading, opencv-python for some image processing (resizing, augmentations), and imgaug for augmentations like flips and brightness shifts.

For modeling, we primarily used Keras and TensorFlow as underlying engines for Mask R-CNN (the Matterport implementation) and a bit of Darknet (via Darkflow) for YOLO. Infrastructure: Model training was done on a GPU (NVIDIA Tesla K80) provided via Kaggle Kernels and Google Colab for convenience.

We had to handle file I/O carefully because the dataset was large (several GB of images). We used Google Drive to store intermediate preprocessed data when using Colab.

Development pipeline: Because training detection can be slow, we adopted a strategy of prototyping on a small subset (like 1000 images) to verify code, then scaling up. We also used the challenge's two-stage approach (Stage 1, Stage 2) – our pipeline allowed us to easily re-run on the Stage 2 test set when it was released by just pointing to new image files.

APIs and Reproducibility: We leveraged Kaggle's API to download the data and sample submission files. For final submission, we wrote a script to run our model on test images and output a CSV of predicted boxes (with coordinates and confidence), matching the required format.

Conclusion and Lessons Learned Outcome: Our project successfully built a prototype pneumonia detection model that can mark suspected pneumonia on chest X-rays with reasonable accuracy. While not yet at radiologist level, it demonstrated the potential of deep learning in medical imaging.

In the RSNA Kaggle challenge, our approach would rank somewhere in the middle of the leaderboard, reflecting there was room to improve especially by ensembling and tuning. Key Lessons: Data quality and labeling matter: We realized some "misses" were due to ambiguous labels.

For instance, if an image was labeled no pneumonia but had an opacity (could be another issue), the model got confused. In medical datasets, labels can be noisy.

Engaging domain experts to refine labels or using semi-supervised methods could help. Transfer learning was crucial: Starting from ImageNet weights or COCO detection weights gave a huge boost.

Training from scratch on ~10k images would likely have failed. The pre-trained backbone had learned to recognize edges, textures, etc., which transferred to recognizing opacities.

Balancing sensitivity and specificity: In medical tasks, there's often a trade- off. We leaned towards high sensitivity – our model catches most pneumonias at the expense of some false alarms.

This is usually acceptable in screening (radiologist can double-check false positives). We learned how to adjust model thresholds and outputs to meet a target operating point (for example, if a hospital says "no more than 10% false positives to avoid overburdening doctors", we could tune to that).

Model explainability: The bounding box output itself is a form of explanation (showing where the pneumonia might be). This is much better than a black-box classification.

We found that presenting the model's output visually to clinicians earned more trust – a critical lesson in AI for healthcare: always provide interpretable results. We also learned to generate heatmaps and discuss false positives/negatives with potential end-users in mind.

## Methodology

Overall, this project solidified our skills in computer vision, particularly in the medical domain. We navigated challenges of data preprocessing, class imbalance, and the critical need for model transparency.

The encouraging results underscore the value of machine learning in aiding disease detection, while the challenges remind us that such tools are to assist, not replace, human expertise (at least for now). It was a rewarding experience to apply deep learning to a problem with direct real-world impact potential.

## Future Work

Complex models and training time: Training Mask R-CNN was non-trivial; with limited GPU, each epoch took time. We learned to optimize by freezing certain layers initially and then fine-tuning, to get results quicker.

Also, detection frameworks have many hyperparameters (anchors, etc.), which we tuned based on validation feedback. This taught us patience and systematic tuning (changing one thing at a time).

Evaluation beyond single metric: We didn't just rely on mAP; we also considered recall, precision, and the clinical relevance of errors. This multi- faceted evaluation is important for a well-rounded understanding.

For example, an overall mAP might be low, but if almost all high-severity pneumonias are caught, the system could still be useful in practice. Thus, we dissected performance by pneumonia size/location and found the model did better on large, obvious pneumonias and worse on faint, small ones – knowledge that could guide future data augmentation (maybe synthesize subtle cases) or model design.

Next steps: To further improve, we'd look into: Ensembling multiple model architectures (each might catch different cases). Utilizing the fact that many chest X-rays come with radiology reports (not in this dataset, but generally) – possibly doing a multi-modal approach or using weakly supervised learning on a larger dataset with report-text.

Trying newer architectures like EfficientDet or a customized U-Net for segmentation of opacities (segmentation might delineate pneumonia better than just boxes).

## References

- [1] [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge).
- [2] [He, K. et al. Deep Residual Learning for Image Recognition.](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) CVPR 2016.
- [3] [Pedregosa, F. et al. Scikit-learn: Machine Learning in Python.](https://jmlr.org/papers/v12/pedregosa11a.html) JMLR 2011.
