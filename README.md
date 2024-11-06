# Automated Knee Arthritis Severity Classification Using X-ray Images

## Project Overview
Knee arthritis, a common joint disease, leads to cartilage breakdown, pain, and disability. This project focuses on automating the detection and classification of knee arthritis severity using X-ray images. Leveraging advanced feature extraction and Convolutional Neural Networks (CNNs), we aim to classify the stages of knee arthritis and improve diagnostic efficiency in medical imaging.

## Data Preparation

### Dataset Link: https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity

### Dataset Understanding
The dataset initially contains five classes of knee osteoarthritis severity, with the following distribution:

| Class      | Label | Image Count |
|------------|-------|-------------|
| Doubtful   | 1     | 1,046       |
| Healthy    | 0     | 2,286       |
| Minimal    | 2     | 1,516       |
| Moderate   | 3     | 757         |
| Severe     | 4     | 173         |

### Data Selection and Augmentation
To create a balanced dataset, we focused on Classes 0, 3, and 4 (Healthy, Moderate, Severe). Data augmentation was used to ensure uniform representation:

| Class    | Label | Image Count |
|----------|-------|-------------|
| Healthy  | 0     | 500         |
| Moderate | 3     | 500         |
| Severe   | 4     | 500         |

### Data Preprocessing
We applied the following preprocessing steps:
- **Resize**: Images resized to 224x224 pixels.
- **Normalization**: Normalized using ImageNet's mean and standard deviation to enhance model performance and stability.

## Model Training

### Model Architecture: Cascade Model
A cascading architecture was employed, leveraging EfficientNet, ResNet, and DenseNet for robust multi-class classification. The model processes images in the following order:
1. **EfficientNet** classifies images initially.
2. If the EfficientNet prediction is incorrect, the image is passed to **ResNet**.
3. For further incorrect predictions, **DenseNet** is used as a final classifier.

### Training Loop
- **Criterion**: Cross-entropy loss.
- **Optimizer**: Adam with weight decay for regularization.
- **Early Stopping**: Implemented based on validation accuracy with patience for optimal model selection.

## Model Evaluation and Metrics

### Training Results
The model training included 10 epochs, with early stopping to avoid overfitting. Here is a sample of the training and validation accuracy over epochs:
- **Epoch 1**: Training Accuracy: 70.07%, Validation Accuracy: 65.08%
- **Epoch 4**: Training Accuracy: 99.93%, Validation Accuracy: 91.97%
- **Early Stop**: Triggered after epoch 7

### Test Results
After training, the model achieved a test accuracy of **92.33%**, indicating effective classification of knee arthritis severity.

## Conclusion
This project successfully implemented a robust, automated method for classifying knee arthritis severity from X-ray images. By leveraging multiple CNN architectures in a cascading model, we achieved high accuracy.
