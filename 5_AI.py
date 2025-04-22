# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd 
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# # from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neighbors import KNeighborsClassifier
# # iris = load_iris()
# # df = pd.DataFrame(data= iris.data, columns=iris.feature_names)
# # df['target'] = iris.target
# # print(df.shape)
# # X_Train, X_Test, y_Train, y_Test = train_test_split(df[iris.feature_names], df['target'], test_size=0.15, random_state=42)

# # # X_Val, X_Test, y_Val, y_Test = train_test_split(X_Test, y_Test)

# # # print(df)

# df = sns.load_dataset('titanic')
# df.drop('deck',axis=1, inplace=True)
# df.dropna(inplace=True)
# df = df.select_dtypes(include=["float64", "int64"])
# X_Train, X_Test, y_Train, y_Test = train_test_split(df.drop('survived',axis=1), df['survived'], test_size=0.15, random_state=42)

# scaler = StandardScaler()
# X_Train = scaler.fit_transform(X_Train)
# X_Test = scaler.transform(X_Test)
# print(X_Train)

# model = LogisticRegression()
# model.fit(X_Train, y_Train) # fit = train

# y_pred = model.predict(X_Test)

# print(accuracy_score(y_Test,y_pred))

# knn = KNeighborsClassifier(5) # k - 5

# knn.fit(X_Train, y_Train)
# y_pred_knn = knn.predict(X_Test)


# print(accuracy_score(y_Test,y_pred_knn))

# # mano_pred = model.predict([ 2  27.0      1      0   49.3000]) # Nusistatyti ar jūs būtumėte išgyvenes titaniko nelaimę
# from sklearn.preprocessing import MinMaxScaler # normalizacija o standartScaler standartizacija
# [].del
