Lung Cancer Classification Pipeline: -

1. Data Acquisition:
*   Data Source -> https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images/data
*   Data Consists of 5000 image, after Augmentation 15000.
---------------------------------------------------------------------
2.  Data Preprocessing:
*   Resizing images to 128,128.
*   32 image per batch.
*   Scaling images.
*   Augmentation Using ImageDataGenerator
---------------------------------------------------------------------
3. Model Building:
*   Consisted of 3 Convolutional, Pooling Layers
*   Flatten & 2 Dense Layers with activation relu
*   Output Layer softmax activation.
*   Model Trained for 50 epoch.
*   Applied early stopping callback with patience 3.
*   Stopped at epoch 16, best weight is epoch 13
---------------------------------------------------------------------
4. Model Evaluation:
*   Model performed 0.95 val_accracy, 0.1 val_loss
*   True Predictions for Benign & squamous.
*   1 False Prediction in adenocarcinoma class.
---------------------------------------------------------------------
