# Deep Learning Project - Multilabel Classification

## Project Description
This project aims to create a multilabel classification model using satellite images. Each image is associated with multiple labels corresponding to different land-use types.

## Dataset
The dataset used consists of satellite images labeled with several categories. The dataset is split into three parts:
- **Training**: 75% of the images
- **Validation**: 10% of the images
- **Test**: 15% of the images

The images are resized to 227x227 pixels, and random transformations (rotation, flipping, color adjustment) are applied to enhance the model's robustness.

## Models Tested
Several models were tested during the project:
1. **AlexNet**
    - This pioneering model was chosen for its simplicity.
    - Global Accuracy: **90.85%**
    - Hamming Loss: **9.15%**

2. **ResNet-18**
    - Selected after testing several architectures. It proved more efficient due to its residual blocks.
    - Global Accuracy: **92.49%**
    - Hamming Loss: **7.51%**

## Model Results

| Model    | Global Accuracy    | Hamming Loss |
|----------|--------------------|--------------|
| AlexNet  | 90.85%             | 9.15%        |
| ResNet-18| 92.49%             | 7.51%        |

### Class-wise Comparison
ResNet-18 showed better overall performance across several classes compared to AlexNet. For example:
- **airplane**: Precision of 86.67%, Recall of 92.86% with ResNet-18, compared to 100% and 75% with AlexNet.
- **cars**: Precision of 80.12%, Recall of 92.14% with ResNet-18, compared to 68.72% and 94.62% with AlexNet.

## Model Workflow

### Data Preparation
The images are transformed and resized before being fed into the model. DataLoaders are configured to load images in batches of 32.

### Model Training
The model uses **BCELoss** as the loss function to compute the error and **Adam** as the optimizer with an initial learning rate of 0.0001. An **EarlyStopping** mechanism is implemented to avoid overfitting.

### Validation and Testing
The model is regularly evaluated on the validation set to monitor performance, and training is stopped when no further improvement is observed.

### Final Model Selection
Ultimately, **ResNet-18** was selected as the main model for its superior ability to generalize to the data.

## Conclusion
The **ResNet-18** architecture demonstrated better global accuracy and more effective handling of under-represented classes compared to AlexNet. While AlexNet is simpler, ResNet-18 proved to be more robust and performant for the task of multilabel classification of satellite images.

## Useful Links
- [CNN Architecture Comparison](https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96)

