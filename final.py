import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('diabetes-dataset.csv')

# Handling missing values
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())
data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())

# Split the data into features (X) and target (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_params(dim):
    theta = np.zeros((dim, 1))
    return theta

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5  # Small constant to prevent log(0)
    cost = (1 / m) * (-y.T @ np.log(h + epsilon) - (1 - y).T @ np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []
    
    for _ in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta -= alpha * gradient
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

# Add a column of ones for the bias term
X = np.c_[np.ones((X.shape[0], 1)), X]


# Initialize parameters and hyperparameters
theta = initialize_params(X.shape[1])
alpha = 0.01
num_iterations = 1000

# Perform gradient descent to train the logistic regression model
optimal_theta, cost_history = gradient_descent(X, y.values.reshape(-1, 1), theta, alpha, num_iterations)

# Function to make predictions
def predict(X, theta):
    probabilities = sigmoid(X @ theta)
    return (probabilities >= 0.5).astype(int)

# Split the data into train and test sets
split_ratio = 0.8
split_index = int(split_ratio * len(data))

X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]

# Make predictions on the test set
y_pred = predict(X_test, optimal_theta)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test.values.reshape(-1, 1)) * 100
print(f"Logistic Regression Test Accuracy: {accuracy:.2f}%")

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of a numerical feature
plt.figure(figsize=(8, 6))
sns.histplot(data['Glucose'], bins=10, kde=True)
plt.xlabel('Glucose')
plt.ylabel('Count')
plt.title('Distribution of Glucose')
plt.show()

# Bar plot of a categorical feature
plt.figure(figsize=(8, 6))
sns.countplot(data['Outcome'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Distribution of Outcome')
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='RdYlBu')
plt.title('Correlation Matrix')
plt.show()

# Scatter plot of two numerical features



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,
                                               random_state=42)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
pipeline_lr  = Pipeline([('scalar1',StandardScaler()),
                         ('lr_classifier',LogisticRegression())])

pipeline_knn = Pipeline([('scalar2',StandardScaler()),
                          ('knn_classifier',KNeighborsClassifier())])

pipeline_svc = Pipeline([('scalar3',StandardScaler()),
                         ('svc_classifier',SVC())])
pipeline_rf = Pipeline([('rf_classifier',RandomForestClassifier(max_depth=9))])
pipeline_gbc = Pipeline([('gbc_classifier',GradientBoostingClassifier())])
pipelines = [pipeline_lr,
            pipeline_knn,
            pipeline_svc,
            pipeline_rf,
            pipeline_gbc]
pipelines

for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_dict = {0:'LR',
             1:'KNN',
             2:'SVC',
             3: 'RF',
             4: 'GBC'}
pipe_dict

for i,model in enumerate(pipelines):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(X_test,y_test)*100))

from sklearn.ensemble import RandomForestClassifier
X = data.drop('Outcome',axis=1)
y = data['Outcome']
rf =RandomForestClassifier(max_depth=3)
rf.fit(X,y)
"""Prediction on New DATA"""
new_data = pd.DataFrame({
    'Pregnancies':6,
    'Glucose':148.0,
    'BloodPressure':72.0,
    'SkinThickness':35.0,
    'Insulin':79.799479,
    'BMI':33.6,
    'DiabetesPedigreeFunction':0.627,
    'Age':50,    
},index=[0])

p = rf.predict(new_data)
if p[0] == 0:
    print('non-diabetic')
else:
    print('diabetic')

"""Save Model Using Joblib"""
import joblib
joblib.dump(rf,'model_joblib_diabetes')
model = joblib.load('model_joblib_diabetes')
model.predict(new_data)


#GUI
'''
from tkinter import *
from pillow import ImageTk, Image
import joblib
import numpy as np
from sklearn import *

master = Tk()

def show_entry_fields():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())

    model = joblib.load('model_joblib_diabetes')
    result = model.predict([[p1, p2, p3, p4, p5, p6, p7, p8]])

    if result == 0:
        Label(master, text="Non-Diabetic").place(x=1700, y=550)
    else:
        Label(master, text="Diabetic").place(x=1700, y=550)

master.title("Diabetes Prediction Using Machine Learning")

# Set the window size to full screen
width, height = master.winfo_screenwidth(), master.winfo_screenheight()
master.geometry("%dx%d+0+0" % (width, height))

# Load and display the image
image = Image.open("diabetes.png")
image = image.resize((width, height), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image)
image_label = Label(master, image=photo)
image_label.place(x=0, y=0)

label = Label(master, text="Diabetes Prediction Using Machine Learning",
              bg="black", fg="white", font=("Arial", 16, "bold"))
label.place(x=800, y=50)

Label(master, text="Pregnancies").place(x=1350, y=150)
Label(master, text="Glucose").place(x=1350, y=200)
Label(master, text="Enter Value of BloodPressure").place(x=1350, y=250)
Label(master, text="Enter Value of SkinThickness").place(x=1350, y=300)
Label(master, text="Enter Value of Insulin").place(x=1350, y=350)
Label(master, text="Enter Value of BMI").place(x=1350, y=400)
Label(master, text="Enter Value of DiabetesPedigreeFunction").place(x=1350, y=450)
Label(master, text="Enter Value of Age").place(x=1350, y=500)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)

e1.place(x=1700, y=150)
e2.place(x=1700, y=200)
e3.place(x=1700, y=250)
e4.place(x=1700, y=300)
e5.place(x=1700, y=350)
e6.place(x=1700, y=400)
e7.place(x=1700, y=450)
e8.place(x=1700, y=500)

Button(master, text='Predict', command=show_entry_fields).place(x=1550, y=550)

# Center all the rows
for i in range(1, 9):
    master.grid_rowconfigure(i, weight=1)

mainloop()


import matplotlib.pyplot as plt

accuracy_scores = []

for i, model in enumerate(pipelines):
    accuracy = model.score(X_test, y_test) * 100
    accuracy_scores.append(accuracy)

plt.bar(pipe_dict.values(), accuracy_scores)
plt.xlabel('Models')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy for Different Models')
plt.show()
'''
import tkinter as tk
from tkinter import Label, Entry, Button
from PIL import Image, ImageTk  # Import PIL for working with images
import joblib
import numpy as np

# Create the main application window
master = tk.Tk()
master.title("Diabetes Prediction Using Machine Learning")

# Set the window size to full screen
width, height = master.winfo_screenwidth(), master.winfo_screenheight()
master.geometry("%dx%d+0+0" % (width, height))

# Load and display the image
image = Image.open("diabetes.png")
image = image.resize((width, height), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image)
image_label = Label(master, image=photo)
image_label.place(x=0, y=0)

# ... rest of your code ...

# Start the main event loop
master.mainloop()
