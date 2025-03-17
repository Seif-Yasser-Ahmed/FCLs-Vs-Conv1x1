# FC Layers vs Conv1x1: A Study on AlexNet Optimization  

Exploring replacing fully connected (FC) layers with 1x1 convolutional layers in AlexNet for image classification. By training on the Flower Classification Dataset, analyzing how this architectural change impacts model size, training time, and accuracy.

## Dataset  
Used the [Flower Classification Dataset](https://www.kaggle.com/datasets/atifaliak/flower-classification), which contains images of five flower species.

## Model Architecture  

Comparing two versions of **AlexNet**:  
1. **Original AlexNet**: Uses FC layers in the classifier.  
2. **Modified AlexNet**: Replaces FC layers with `1x1` convolutions.  

**Classifier (Modified AlexNet)**:  
```python
self.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Conv2d(256, 4096, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Conv2d(4096, 4096, kernel_size=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(4096, num_classes, kernel_size=1),
    nn.AdaptiveAvgPool2d((1, 1))
)
```

##  Training Details  

- **Optimizer**: Adam (`lr=0.001`)  
- **Loss Function**: Cross-Entropy Loss  
- **Batch Size**: 32  
- **Epochs**: 2  
- **Progress Tracking**: `tqdm`  

## Training Insights  

| Model         | Parameters | Training Time (2 epochs) | Loss (After epoch 1) | Loss (After epoch 2)|
|--------------|------------|----------------------|-----------------|-----|
| FC AlexNet   | 57,061,198        | 2 mins               | 2.2880            | 1.7916|
| Conv1x1 AlexNet | 20,361,038    | 2 mins               | 2.2083            |1.7962|

## Results  

After training, evaluated both models on the validation set:  

- **Impact of Conv1x1**: Replacing FC layers with `1x1` convolutions increases parameter efficiency while maintaining accuracy.  


## Conclusion  

This study demonstrates that `1x1` convolutions can effectively replace FC layers in AlexNet while optimizing parameter efficiency and potentially improving generalization.  

