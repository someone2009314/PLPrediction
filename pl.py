import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

file_path = r"C:\Users\maste\OneDrive\Desktop\AI\Premier League\premier-league-tables.csv"

df = pd.read_csv(file_path, delimiter=';')

df = df.iloc[:, 0].str.split(',', expand=True)
df.drop(df.columns[12], axis=1, inplace=True) #Making 12 columns
df.columns=['Season_End_Year', 'Team', 'Rk', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Notes']

#Dropping notes column and turning team column into numbers
le = LabelEncoder()
df['Team'] = le.fit_transform(df['Team'])
df.drop('Notes', axis = 1, inplace=True)

#X is equal to all the columns except points
X = df.drop(columns=['Pts'])
#y is equal to only pts because it is the output
y = df['Pts']

X=X.to_numpy()
y=y.to_numpy()

#X=X.reshape(-1, 10, 1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

#RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)


model.fit(X_train, y_train)

#Prediction and accuracy
y_pred = model.predict(X_test)

print(mean_squared_error(y_pred, y_test))
print(r2_score(y_pred, y_test))

