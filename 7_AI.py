
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier


df = sns.load_dataset('titanic')
df.drop('deck',axis=1, inplace=True)
df.drop('alive',axis=1, inplace=True)
df.drop('embark_town',axis=1, inplace=True)
df.drop('class',axis=1, inplace=True)
df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True) # ONE HOT ENCODING

X_Train, X_Test, y_Train, y_Test = train_test_split(df.drop('survived',axis=1), df['survived'], test_size=0.15, random_state=42)


# model = RandomForestClassifier(random_state=42) # 100 trees

# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [50, 100, 200, 300, 500],
#     'max_depth': [2,5,8, 10, 20],
#     'min_samples_split': [2, 5, 10,20,30,40]
# }

# gridcv = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=3)

# gridcv.fit(X_Train, y_Train)
# print(gridcv.best_params_)
# print(gridcv.best_score_)
# model = gridcv.best_estimator_

# y_pred = model.predict(X_Test)
# print(accuracy_score(y_Test,y_pred))

# vs boost
from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(max_depth=2, random_state=42)

# model = AdaBoostClassifier(estimator=base_estimator, n_estimators=1000, random_state=42)
# model = GradientBoostingClassifier(random_state=42, verbose=3, min_samples_split=2, max_depth=20)
model = HistGradientBoostingClassifier(random_state=42, verbose=1)

param_grid = {
    'max_depth': [1,2,3,4,5,7,9,10],
    "min_samples_leaf": [2, 5, 10,20,30,40],
    'max_iter': [100, 200, 300, 500],
    'learning_rate': [0.001,0.01, 0.1, 1, 10]
}
from sklearn.model_selection import GridSearchCV

gridcv = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=3)
gridcv.fit(X_Train, y_Train)
print(gridcv.best_params_)
model = gridcv.best_estimator_

y_pred = model.predict(X_Test)
print(accuracy_score(y_Test,y_pred))