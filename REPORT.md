# Synthetic Shape Detection Report Draft

## 1. Introduction

- Course: `CENG428 Neural Networks`
- Task: `Synthetic Shape Detection on MS COCO 2017`
- Formulation: `Object Detection`
- Main framework: `PyTorch`

## 2. Dataset Preparation

Describe:

- where `MS COCO 2017` was placed
- how `torchvision.datasets.CocoDetection` was used
- why COCO labels were not used as target labels
- how natural images were used only as backgrounds

Draft notes:

- COCO `train2017` and `val2017` were used as required by the assignment.
- The dataset was loaded with `CocoDetection`.
- Original COCO object annotations were not used as training targets.
- Synthetic shapes were added on top of COCO images, and the labels came from those synthetic additions.

## 3. Train / Validation / Test Split

Describe:

- sorting COCO image ids
- first `5000` images from `train2017` for training
- first `1000` images from `val2017` for validation
- next `1000` images from `val2017` for testing
- deterministic generation for validation and test with `GLOBAL_SEED = 2025`

Draft notes:

- Validation and test shape generation were made reproducible with a stable hash-based seed function.

## 4. Synthetic Shape Generation

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

## 5. Label Generation Method

Describe:

- how bounding boxes were produced automatically during shape drawing
- how empty targets were represented for negative images
- the label convention for the detector

Draft notes:

- All synthetic shapes can share one foreground class label, such as `1`.
- Negative images should return empty `boxes` and empty `labels` tensors with valid shapes.

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
