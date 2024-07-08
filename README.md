# Breast Cancer Prediction Using Machine Learning

This project demonstrates how to build a logistic regression classifier to predict breast cancer diagnosis based on the Wisconsin Breast Cancer dataset. It includes data preprocessing, model training, and evaluation with a confusion matrix and accuracy score.

## Introduction

Breast cancer is one of the most common cancers among women worldwide. Early detection and diagnosis are crucial for effective treatment and improved survival rates. Machine learning techniques can assist in predicting breast cancer diagnosis based on features extracted from diagnostic tests.

## Dataset

The dataset used in this project is the Wisconsin Breast Cancer dataset (`data/data.csv`). It includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension, and the target variable indicating the diagnosis (M = malignant, B = benign).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/breast-cancer-prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd breast-cancer-prediction
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place your `data.csv` file in the `data/` directory.
2. Run the main script:
    ```sh
    python src/main.py
    ```

## Project Structure

- `data/`: Contains the dataset.
- `src/`: Contains the main script.
- `.gitignore`: Specifies files to be ignored by Git.
- `README.md`: Project documentation.
- `requirements.txt`: List of Python packages required to run the project.

## Results

After running the script, the model evaluates its performance using a confusion matrix and calculates the accuracy score. The confusion matrix visualizes the true positive, true negative, false positive, and false negative predictions, providing insights into the model's effectiveness in diagnosing breast cancer.

## Requirements

- pandas
- scikit-learn
- seaborn
- matplotlib

## Future Improvements

In the future, additional machine learning algorithms such as support vector machines (SVMs) or neural networks could be explored to further enhance prediction accuracy. Furthermore, integrating more diverse datasets and implementing cross-validation techniques could improve the robustness of the model.

---

