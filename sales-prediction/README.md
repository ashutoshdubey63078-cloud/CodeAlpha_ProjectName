# Sales Prediction Project

## Overview
The Sales Prediction project aims to predict sales based on advertising spend across different media channels. This project utilizes machine learning models to analyze the relationship between advertising expenditures and sales performance. The primary model used for predictions is a Random Forest Regressor, which has been trained on a dataset containing various features related to advertising.

## Dataset
The dataset used for this project is located in the `dataset` directory and is named `advertising.csv`. It includes the following features:
- TV: Advertising spend on television
- Radio: Advertising spend on radio
- Newspaper: Advertising spend on newspapers
- Sales: Sales generated from the advertising spend

## Installation Instructions
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd sales-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To use the trained model for sales prediction, you can load the model from the `models` directory and input your advertising spend data. The model will return the predicted sales value.

### Example
```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('models/sales_prediction_model.pkl')

# Sample input data
sample = pd.DataFrame({
    "TV": [230.1],
    "Radio": [37.8],
    "Newspaper": [69.2]
})

# Make a prediction
prediction = model.predict(sample)
print("Predicted Sales:", prediction[0])
```

## Jupyter Notebook
The project includes a Jupyter notebook located in the `notebooks` directory named `Sales_Prediction.ipynb`. This notebook contains the complete workflow for data loading, preprocessing, exploratory data analysis, model training, evaluation, and visualization of model performance.

## Model
The trained model is saved in the `models` directory as `sales_prediction_model.pkl`. This file contains the Random Forest model that has been trained on the dataset.

## Requirements
The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

These dependencies are listed in the `requirements.txt` file.

## Conclusion
This project demonstrates the application of machine learning techniques for sales prediction based on advertising spend. The provided notebook and model can be used as a foundation for further exploration and enhancement of predictive analytics in marketing.