
#import libraries

import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#import dataset

file_path = r'C:\Users\tinao\OneDrive\Radna površina\data mining project\dataset_heart.csv'  # r is there to prevent escape sequence mistakes
df = pd.read_csv(file_path)

df.head()

# Check for missing values
missing_values = df.isnull().sum().sum()
print("Number of missing values: " + str(missing_values))

print(df.dtypes)

df.describe().T

#check for duplicates
duplicates = df.duplicated().sum()
print("Number of duplicate values: " + str(duplicates))

# make boxplots for every attribute
def boxplots(df):
    cols = df.columns[:-1]
    n = (len(cols) - 1) // 6 + 1
    m = min(len(cols), 6)
    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(15, 10))
    for idx, col in enumerate(cols):
        i = idx // m
        j = idx % m
        sns.boxplot(data=df, x=col, ax=axes[i][j])

    plt.tight_layout()
    plt.show()

boxplots(df)


# Explore the distribution of the target variable
sns.countplot(x='heart disease', data=df)
plt.title('Distribution of Heart Attack')
plt.show()

plt.figure(figsize=(14, 12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='Blues') #or Purples
plt.title('Correlation Matrix')
plt.show()

#fix outliers

def outliers_removal(df, i):
    perc = np.percentile(df[i], [0, 25, 50, 75, 100])
    iqr = perc[3] - perc[1]
    _min = perc[1] - 1.5*iqr
    _max = perc[3] + 1.5*iqr

    # Check if the column is of integer type, and cast the min and max values accordingly
    if pd.api.types.is_integer_dtype(df[i]):
        _min = int(_min)
        _max = int(_max)

    df.loc[df[i] > _max, i] = _max
    df.loc[df[i] < _min, i] = _min
    return df

for i in df.columns[:-1]:
    df = outliers_removal(df, i)

    boxplots(df)

# Pair plot
sns.pairplot(df)
plt.show()

# Split the data into features (X) and target variable (y)
X = df.drop('heart disease', axis=1)
y = df['heart disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
df2 = df.copy()
ss = StandardScaler()
df2[['age', 'resting blood pressure','serum cholestoral', 'max heart rate','oldpeak']] = ss.fit_transform(df2[['age','resting blood pressure','serum cholestoral', 'max heart rate','oldpeak']])

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# improt ALl models.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
#importing pipeline
from sklearn.pipeline import Pipeline

# import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error

print("works fine.")

#Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier




classifier= LogisticRegression(random_state=0)
regressor= LinearRegression()
dt = DecisionTreeClassifier(max_depth=5)


classifier.fit(X_train, y_train)

regressor.fit(X_train, y_train)

dt.fit(X_train, y_train)

#Predicting the test set result
#y_pred= classifier.predict(X_test)


from sklearn.metrics import accuracy_score

y_pred = classifier.predict(X_test)
print('Logistic regression  Test Accuracy ', accuracy_score(y_test, y_pred ))

y_pred = dt.predict(X_test)
print('Decision Tree Test Accuracy ', accuracy_score(y_test, y_pred ))

#y_pred = regressor.predict(X_test)
#print('Linear regression Test Accuracy ', accuracy_score(y_test, y_pred ))

y_pred_regressor = regressor.predict(X_test)
print('Linear Regression Test MSE:', mean_squared_error(y_test, y_pred_regressor))

from sklearn.metrics import classification_report

def plot_classification_report(y_train, y_pred1, y_test, y_pred2, c_name):
    print("-"*25,c_name,"(TRAIN SET)","-"*25)
    print(classification_report(y_train, y_pred1))
    print("-"*25,c_name,"(Test SET)","-"*25)
    print(classification_report(y_test, y_pred2))


c_name= "Logistic Regression"
plot_classification_report(y_train, classifier.predict(X_train), y_test, classifier.predict(X_test), c_name)

c_name= "Decision Tree"
plot_classification_report(y_train, dt.predict(X_train), y_test, dt.predict(X_test), c_name)

#c_name= "Linear Regression"
#plot_classification_report(y_train, regressor.predict(X_train), y_test, regressor.predict(X_test), c_name)

from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def plot_model_report(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name, is_classification):
    if is_classification:
        print(f"{model_name} Classification Report for Training Data:\n")
        print(classification_report(y_train_true, y_train_pred))

        print(f"{model_name} Classification Report for Test Data:\n")
        print(classification_report(y_test_true, y_test_pred))

        # Additional plots for classification (e.g., confusion matrix) can be added here
    else:
        print(f"{model_name} Regression Report for Training Data:\n")
        print(f"Mean Squared Error: {mean_squared_error(y_train_true, y_train_pred)}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_train_true, y_train_pred)}")
        print(f"R-squared: {r2_score(y_train_true, y_train_pred)}")

        print(f"{model_name} Regression Report for Test Data:\n")
        print(f"Mean Squared Error: {mean_squared_error(y_test_true, y_test_pred)}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test_true, y_test_pred)}")
        print(f"R-squared: {r2_score(y_test_true, y_test_pred)}")

        # Plotting true vs predicted values for test data
        plt.scatter(y_test_true, y_test_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} True vs Predicted Values')
        plt.show()


# Creating the Confusion matrix
# from sklearn.metrics import confusion_matrix
# cm= confusion_matrix(y_test, y_pred)

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_train_true, y_train_pred, y_test_true, y_test_pred, classifier_name):
    train_cm = confusion_matrix(y_train_true, y_train_pred)
    test_cm = confusion_matrix(y_test_true, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Train Confusion Matrix
    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f'Train Confusion Matrix - {classifier_name}')
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')

    # Test Confusion Matrix
    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title(f'Test Confusion Matrix - {classifier_name}')
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('True Labels')

    plt.show()

# Generate confusion matrix for each classifier
plot_confusion_matrix(y_train, classifier.predict(X_train), y_test, classifier.predict(X_test), "Logistic Regression")
#plot_confusion_matrix(y_train, dt.predict(X_train), y_test, dt.predict(X_test), "Decision Tree")

plot_confusion_matrix(y_train, dt.predict(X_train), y_test, dt.predict(X_test), "Decision Tree")

c_name = "Linear Regression"
is_classification = False

plot_model_report(y_train, regressor.predict(X_train), y_test, regressor.predict(X_test), c_name, is_classification)



