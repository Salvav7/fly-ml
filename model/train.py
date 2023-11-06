import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import dump



flights_jan_2020 = pd.read_csv("data/Jan_2020_ontime.csv")
# Data pre-processing


# Select relevant features
features = ['DAY_OF_WEEK', 'DEP_DEL15', 'ARR_DEL15']

flights_jan_2020 = flights_jan_2020[features].dropna()

# Split the data into training and testing sets. 

X = flights_jan_2020.drop('ARR_DEL15', axis=1)
y = flights_jan_2020['ARR_DEL15']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a Random Forest classifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

dump(model, 'model/flights-jan-v1.joblib')