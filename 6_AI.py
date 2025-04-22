# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split, cross_val_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd 
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.neighbors import KNeighborsClassifier


# df = sns.load_dataset('titanic')
# df.drop('deck',axis=1, inplace=True)
# df.drop('alive',axis=1, inplace=True)
# df.drop('embark_town',axis=1, inplace=True)
# df.drop('class',axis=1, inplace=True)
# df.dropna(inplace=True)

# df = pd.get_dummies(df, drop_first=True) # ONE HOT ENCODING
# print(df.columns)
# X_Train, X_Test, y_Train, y_Test = train_test_split(df.drop('survived',axis=1), df['survived'], test_size=0.15, random_state=42)

# scaler = StandardScaler()
# X_Train = scaler.fit_transform(X_Train)
# X_Test = scaler.transform(X_Test)

# # scaler = MinMaxScaler((0,1))
# # X_Train = scaler.fit_transform(X_Train)
# # X_Test = scaler.transform(X_Test)

# # scaler = RobustScaler()
# # X_Train = scaler.fit_transform(X_Train)
# # X_Test = scaler.transform(X_Test)

# accuracies = []
# for k in range(1,40):

#     knn = KNeighborsClassifier(k) # Negerai k rinkti pagal test tiksluma, reiketu rinkti apgal validacijos tiksluma

#     knn.fit(X_Train, y_Train)
#     y_pred_knn = knn.predict(X_Test)


#     accuracies.append(accuracy_score(y_Test,y_pred_knn))
    


# sns.lineplot(x=range(1,40),y=accuracies)
# plt.show()

# knn = KNeighborsClassifier(9) # Negerai k rinkti pagal test tiksluma, reiketu rinkti apgal validacijos tiksluma

# knn.fit(X_Train, y_Train)
# y_pred_knn = knn.predict(X_Test)

# cm = confusion_matrix(y_Test, y_pred_knn)
# print(cm)
# cr = classification_report(y_Test, y_pred_knn)
# print(cr)



# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split, cross_val_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd 
# import numpy as np
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.neighbors import KNeighborsClassifier


# df = sns.load_dataset('titanic')
# df.drop('deck',axis=1, inplace=True)
# df.drop('alive',axis=1, inplace=True)
# df.drop('embark_town',axis=1, inplace=True)
# df.drop('class',axis=1, inplace=True)
# df.dropna(inplace=True)

# df = pd.get_dummies(df, drop_first=True) # ONE HOT ENCODING

# X_Train, X_Test, y_Train, y_Test = train_test_split(df.drop('survived',axis=1), df['survived'], test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_Train = scaler.fit_transform(X_Train)
# # X_Val = scaler.transform(X_Val)
# X_Test = scaler.transform(X_Test)

# scaler = MinMaxScaler((0,1))
# X_Train = scaler.fit_transform(X_Train)
# X_Test = scaler.transform(X_Test)

# scaler = RobustScaler()
# X_Train = scaler.fit_transform(X_Train)
# X_Test = scaler.transform(X_Test)

# cv_accuracies = []
# k_values = range(1,60)
# for k in k_values:

#     knn = KNeighborsClassifier(k) # Negerai k rinkti pagal test tiksluma, reiketu rinkti apgal validacijos tiksluma

#     knn.fit(X_Train, y_Train)
#     y_pred_knn = knn.predict(X_Test)

#     scores = cross_val_score(knn, X_Train, y_Train, cv=5, scoring='accuracy') # pereina tiek kartu kiek nurodytas cv, vis pasimdamas gabaliuka, validacijai ir atiduoda kiekvieno gabaliuko validacijos tiksluma
#     cv_accuracies.append(scores.mean())
    


# sns.lineplot(x=k_values,y=cv_accuracies)
# plt.show()

# knn = KNeighborsClassifier(np.argmax(cv_accuracies)+1) 

# knn.fit(X_Train, y_Train)
# y_pred_knn = knn.predict(X_Test)

# cm = confusion_matrix(y_Test, y_pred_knn)
# print(cm)
# cr = classification_report(y_Test, y_pred_knn)
# print(cr)


from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC # regresija SVR


df = sns.load_dataset('titanic')
df.drop('deck',axis=1, inplace=True)
df.drop('alive',axis=1, inplace=True)
df.drop('embark_town',axis=1, inplace=True)
df.drop('class',axis=1, inplace=True)
df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True) # ONE HOT ENCODING

X_Train, X_Test, y_Train, y_Test = train_test_split(df.drop('survived',axis=1), df['survived'], test_size=0.2, random_state=42)

scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)

svm_model = SVC(C=300, kernel='rbf', gamma=0.01, random_state=42) # C - reguliuoja kiek modelis gali klysti, kuo didesnis tuo maziau klysta, bet tuo labiau prisitaiko prie duomenu, gamma - reguliuoja kiek toli ieškoti kaimynų, kuo didesnis tuo mažiau kaimynų ieško

svm_model.fit(X_Train, y_Train)

y_pred = svm_model.predict(X_Test)

from sklearn.ensemble import BaggingClassifier

bagging_model = BaggingClassifier(estimator=svm_model, n_estimators=300, random_state=42, max_samples=0.8, max_features=0.8) # max_samples - kiek imti duomenu, max_features - kiek imti features
bagging_model.fit(X_Train, y_Train)
y_pred_bagging = bagging_model.predict(X_Test)

print(accuracy_score(y_Test,y_pred))
print(accuracy_score(y_Test,y_pred_bagging))


# cm = confusion_matrix(y_Test, y_pred)
# print(cm)
# cr = classification_report(y_Test, y_pred)
# print(cr)

# sns.heatmap(cm, annot=True)
# plt.show()



# from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd 
# import numpy as np
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# df = sns.load_dataset('titanic')
# df.drop('deck',axis=1, inplace=True)
# df.drop('alive',axis=1, inplace=True)
# df.drop('embark_town',axis=1, inplace=True)
# df.drop('class',axis=1, inplace=True)
# df.dropna(inplace=True)
# df = pd.get_dummies(df, drop_first=True) # ONE HOT ENCODING

# X_Train, X_Test, y_Train, y_Test = train_test_split(df.drop('survived',axis=1), df['survived'], test_size=0.2, random_state=42)


# model = DecisionTreeClassifier(max_depth=5)

# model.fit(X_Train, y_Train)

# y_pred = model.predict(X_Test)

# plt.figure(figsize=(30, 15)) 
# tree.plot_tree(model, filled=True, feature_names=df.drop('survived',axis=1).columns)
# plt.show()

# print(accuracy_score(y_Test,y_pred))