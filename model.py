from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np

data = pd.read_csv("loan_data.csv")
X = data[['Income','LoanAmount','CreditScore']]

label = LabelEncoder()
data['New_Approved'] = label.fit_transform(data['Approved'])

y = data['New_Approved']

model = RandomForestClassifier()
model.fit(X,y)

with open("model.pkl","wb") as f:
    pickle.dump(model,f)

with open("label_encoder.pkl","wb") as f:
    pickle.dump(label,f)
    