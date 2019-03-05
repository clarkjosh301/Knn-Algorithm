#Basic Knn algorithm, data from https://www.kaggle.com/brynja/wineuci
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
wine_csv = r'C:\Users\hcpl_bel\Downloads\wineuci\Wine.csv'
wine_df = pd.read_csv(wine_csv)
wine_df.columns = ['name', 'alcohol', 'malicacid', 'ash', 'ashcal.', 'magnesium', 'totalphenols', 'flavaniods', 'nonflavphenols', 'proanthins.', 'colorI', 'hue', '?', 'proline']
X = wine_df.drop(['name'], axis=1)
knn = KNeighborsClassifier(n_neighbors=1)
y = wine_df['name']
X_train, x_test, y_train, y_test = train_test_split(
	X, y, random_state=0)
knn.fit(X_train, y_train)
X = X.values
x_new = np.array([[13, 2, 2.4, 15, 100, 2.2, 2.8, .3, 1.5, 4, .93, 2.6, 1000]])
prediction = knn.predict(x_new)
print("predicted target: {}".format(
	y[prediction]))
#prediction=1 types of wine in the dataset are classified as 1, 2, and 3
