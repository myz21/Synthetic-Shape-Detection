# Synthetic Shape Detection Report Draft

## 1. Introduction

- Course: `CENG428 Neural Networks`
- Task: `Synthetic Shape Detection on MS COCO 2017`
- Formulation: `Object Detection`
- Main framework: `PyTorch`

## 2. Dataset Preparation

MS COCO 2017 was used as the required natural-image source dataset. The repository assumes the standard directory structure under `data/coco/`, including `train2017`, `val2017`, and the corresponding `instances_train2017.json` and `instances_val2017.json` annotation files. The dataset was loaded with `torchvision.datasets.CocoDetection`, but the original COCO object categories were not used as training targets. Instead, COCO images served only as natural backgrounds, and the final labels were derived from synthetic shapes added by our own generation pipeline.

## 3. Train / Validation / Test Split

To follow the assignment protocol exactly, COCO image ids were sorted in increasing order before splitting. The first `5000` images from `train2017` were used for training, the first `1000` images from `val2017` were used for validation, and the next `1000` images from `val2017` were used for testing. Validation and test samples were generated deterministically with `GLOBAL_SEED = 2025` and a stable SHA-256 based seed function, so the same image id always produced the same synthetic sample outside the training split.

## 4. Synthetic Shape Generation

```
Describe:

- shape types used
- how location, size, color, opacity, and rotation were varied
- positive / negative image ratio
- how many shapes were added to positive images
- at least four difficulty mechanisms used

Draft notes:

- Positive images contain at least one synthetic shape.
- Negative images contain no target synthetic shape.
- Planned difficulty settings include opacity variation, easy vs hard overlays, and augmentation-related comparisons.
```

Synthetic samples were created by drawing random shapes directly onto COCO background images. The current generator supports rectangles, ellipses, and triangles, and varies location, size, color, opacity, and the number of shapes per image. Positive images contain between `1` and `3` synthetic shapes, while negative images contain no target shape; the positive/negative ratio is set to approximately `70/30` as suggested in the assignment. To keep the task nontrivial, the generator already includes multiple difficulty mechanisms such as random opacity, low-contrast colors sampled from local image content, Gaussian blur on inserted shapes, and additive noise on the final image. Part of the synthetic shape placement and bounding-box design was inspired by prior experience from a Teknofest Aviation AI project. However, for this assignment, all synthetic shapes and their corresponding labels were generated fully automatically, and no manual annotation was used.

## 5. Label Generation Method

Labels were generated automatically during rendering. For each synthetic shape, the sampled placement parameters directly defined a bounding box in `[x_min, y_min, x_max, y_max]` format, and these boxes were collected into the detection target dictionary. All target shapes shared a single foreground label, `1`, while background was handled implicitly by the detector. For negative images, the target returned valid empty tensors for `boxes` and `labels`, which makes the dataset compatible with object detection training code without any manual editing of annotations.

## 6. Model Architecture

Describe:

- chosen detection model
- whether it was trained from scratch or fine-tuned
- which parts were pretrained
- why this model was selected

Draft template:

- Model:
- Backbone:
- Pretrained weights:
- Detection head:
- Number of classes:
- Reason for selection:

## 7. Training Procedure

Fill in:

- input image size:
- batch size:
- number of epochs:
- optimizer:
- learning rate:
- loss function:
- hardware:
- approximate training time:

## 8. Baseline Method

Describe the simple baseline used for comparison.

Possible example:

- edge detection + connected components
- shallow CNN
- simple image-level classifier

Write:

- what the baseline does
- why it is considered simple
- how it was evaluated against the main model

## 9. Evaluation Metrics

For object detection, report at least:

- `Precision@IoU=0.5`
- `Recall@IoU=0.5`
- `F1-score@IoU=0.5`
- mean IoU of matched predictions

Optional:

- `AP`
- `mAP`

## 10. Experimental Results

At least three meaningful experiments are required.

Planned experiment list:

1. High opacity vs low opacity
2. Easy shapes vs hard shapes
3. Small training set vs larger training set
4. With data augmentation vs without data augmentation

For each experiment, include:

- setup
- result table
- short interpretation

### Experiment 1

- Setup:
- Results:
- Interpretation:

### Experiment 2

- Setup:
- Results:
- Interpretation:

### Experiment 3

- Setup:
- Results:
- Interpretation:

### Experiment 4

- Setup:
- Results:
- Interpretation:

## 11. Prediction Visualizations

Add:

- generated synthetic samples
- prediction visualizations on test images
- successful cases
- failure cases
- positive examples
- negative examples

## 12. Failure Cases

Discuss:

- which synthetic additions were hard to detect
- whether low contrast, blur, transparency, or overlap hurt performance
- whether negatives caused false positives

## 13. Limitations and Possible Improvements

Possible points:

- more realistic synthetic generation
- stronger detector
- better augmentation
- more training data
- better baseline
- AP / mAP evaluation

## 14. Conclusion

Summarize:

- the final detection approach
- the most important quantitative result
- what worked well
- what remained difficult for the model
- the main next improvement you would make

## 15. Submission Checklist

- `README.md`
- notebook or source code
- report file
- result tables
- sample visualizations
- prediction visualizations
- failure-case discussion
- environment description
