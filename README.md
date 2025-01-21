# Pharmaceutical Drugs and Vitamins Classification

This project demonstrates the classification of pharmaceutical drugs and vitamins using a convolutional neural network (CNN) based on the MobileNetV2 architecture. It includes data preprocessing, model training, evaluation, and visualization of predictions.

## Project Overview

The code implements the following:
- Loading and preprocessing image data.
- Splitting the dataset into training, validation, and test sets.
- Using MobileNetV2 as a pre-trained base model.
- Adding custom dense layers for classification.
- Training the model with early stopping and checkpointing.
- Evaluating the model's performance.
- Visualizing predictions and model metrics.

## Requirements

Install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

### Key Libraries:
- pandas
- numpy
- matplotlib
- tensorflow
- keras_preprocessing
- sklearn

## Dataset

The dataset is structured as follows:

```
Drug Vision/Data Combined/
    ├── Alaxan/
    │   ├── 00000005.jpg
    │   ├── 00000006.jpg
    │   └── ...
    ├── Bactidol/
    │   ├── 00000000.jpg
    │   ├── 00000001.jpg
    │   └── ...
    └── ...
```

Update the `dataset` variable in the script with the path to your dataset directory.

### Dataset Source

The dataset used in this project can be accessed [here](https://www.kaggle.com/datasets/vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images).

## How to Run

1. Clone this repository and navigate to the project directory.
2. Update the `dataset` path in the script to point to your dataset.
3. Run the script:

```bash
python classification_script.py
```

## Model Architecture

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers:**
  - Dense layers with ReLU activation
  - Dropout for regularization
  - Softmax layer for classification

## Outputs

- Training and validation accuracy and loss graphs.
- Classification report showing precision, recall, and F1-score.
- Visualization of predictions with true and predicted labels.

## Checkpoints

The model's weights are saved during training at the path specified by `checkpoint_path`. This allows restoring the best-performing model for evaluation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to contribute to this project by submitting issues or pull requests.
