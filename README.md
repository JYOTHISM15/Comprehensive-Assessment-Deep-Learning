# Diabetes Progression Prediction using Artificial Neural Networks (ANN)
## Overview
This project aims to model the progression of diabetes using the Diabetes dataset available in the sklearn library. The goal is to develop a machine learning model using an Artificial Neural Network (ANN) that predicts the progression of diabetes in patients based on several independent variables. The model will help healthcare professionals gain insights into the various factors that influence diabetes progression, and potentially contribute to improving treatment plans and preventive measures.

## Objective
The objective of this project is to:

Build an Artificial Neural Network (ANN) model to predict the progression of diabetes based on available features.
Understand how different factors influence diabetes progression.
Evaluate and improve the performance of the ANN model using appropriate techniques.
## Dataset
The dataset used for this project is the Diabetes Dataset from sklearn.datasets. It contains 10 features related to medical and lifestyle factors, along with the target variable representing the progression of diabetes after one year.

### Key Variables in the Dataset:
Features: Age, Sex, BMI, Blood Pressure, Insulin, Diabetes Pedigree Function, etc.
Target variable: Diabetes progression after one year.
## Key Components
### 1. Loading and Preprocessing
Objective: Load the Diabetes dataset, handle missing values, and normalize the features.
Steps:
Loaded the diabetes dataset from sklearn.datasets.
Checked for missing values (none present in this dataset).
Normalized the features using StandardScaler for better model performance.
### 2. Exploratory Data Analysis (EDA)
Objective: Perform EDA to understand the distribution of features and the target variable, and visualize their relationships.
Steps:
Visualized the distribution of the target variable (diabetes progression).
Created pair plots to show the relationships between the features and the target variable.
Explored the statistical properties of the features.
### 3. Building the ANN Model
Objective: Design an artificial neural network (ANN) architecture with at least one hidden layer and appropriate activation functions.
Steps:
Built a neural network with one hidden layer and used ReLU as the activation function for hidden layers and a linear activation function for the output layer.
### 4. Training the ANN Model
Objective: Split the dataset into training and testing sets, and train the model on the training data.
Steps:
Split the data into training (80%) and testing (20%) sets.
Used Mean Squared Error (MSE) as the loss function and Adam optimizer for training the model.
Trained the model for 100 epochs and monitored its performance using validation data.
### 5. Evaluating the Model
Objective: Evaluate the model on the test data and report key performance metrics such as Mean Squared Error (MSE) and R² Score.
Steps:
Evaluated the trained model on the test set.
Reported the performance using MSE and R² score.
### 6. Improving the Model
Objective: Experiment with different architectures, activation functions, and hyperparameters to improve the model’s performance.
Steps:
Experimented with adding more hidden layers, increasing neurons, and changing activation functions (e.g., using tanh or sigmoid).
Tuned the model by adjusting the learning rate and number of epochs.
Evaluated the performance of the improved model and compared it with the initial version.
## Conclusion
This project demonstrated the application of Artificial Neural Networks (ANN) to predict the progression of diabetes. The ANN model showed good potential in understanding how various factors contribute to the progression of the disease. By experimenting with different architectures and hyperparameters, we were able to improve the model's performance, which can help healthcare professionals in making more informed decisions about diabetes treatment and prevention.

## Future Improvements
Use of additional data sources such as medical history or lifestyle information to improve the prediction accuracy.
Experimenting with more advanced deep learning models, such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs), to capture time-series or spatial data patterns in diabetes progression.
