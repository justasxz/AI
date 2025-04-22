# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
# # užsikrausime csv
# initial_df = pd.read_csv(r"data/flight_delays_train.csv")

# # print(initial_df.head())
# # print(initial_df.shape)

# # print(initial_df.describe())
# # # print(initial_df.info()) # nera tusciu reiksmiu del to galima nepildyti nieko
# # print(initial_df.nunique()) # kiek unikaliu reiksmiu
# # print(initial_df[initial_df["DepTime"] > 2400].head(20)) 
# # esu garantuotas, kad laikas neturetu virsyti 2500, nes yra 24 valandos
# df = initial_df[initial_df["DepTime"] < 2400] 
# # print(df.shape)

# le = LabelEncoder()
# df["dep_delayed_15min"] = le.fit_transform(df["dep_delayed_15min"])
# # print(df.head(20))
# # print(df["dep_delayed_15min"].value_counts()) 
# # for column in df.columns:
# #     print(df[column].value_counts()) 

# destination_counts = df["Dest"].value_counts()
# # surandame kur desination maziau nei 10
# # print(destination_counts[destination_counts <= 5].index)
# # atfiltruojame pagal destination
# df = df[~df["Dest"].isin(destination_counts[destination_counts <= 10].index)] # imam tik tuos kuriu indeksu nera tarp destination_counts maziau lygu  nei 5
# df = df[~df["Origin"].isin(destination_counts[destination_counts <= 10].index)] # imam tik tuos kuriu indeksu nera tarp destination_counts maziau lygu  nei 5
# # print(df["Dest"].shape)
# # print(df.nunique())
# # pakeiskime c-10 i 10
# def prepare_data(df):
#     df.replace("c-", '', inplace=True, regex=True) 
#     columns_to_convert = ["Month", "DayofMonth", "DayOfWeek"]
#     for column in columns_to_convert:
#         df[column] = df[column].astype(int)

#     df['Season'] = df['Month'].map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}) 
#     df.drop(['Month','DayofMonth','DayOfWeek'], axis=1, inplace=True) # drop month column

#     return df
# # change type of Month, DayofMonth, DayOfWeek to int

# # correlation matrix
# # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# # sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
# # plt.title("Correlation Matrix")
# # plt.show()
# # Feature engineering (kad mes pasiziureje koreliacija, matome, kad monthi r t.t. turi itakos, mes galime rankiniu budu pakeisti i sezonus, pvz winter summer ir t.t.)
# # mes galime padaryti kad monthas butu sezonas, t.y. 1-3 winter, 4-6 spring, 7-9 summer, 10-12 autumn
# df = prepare_data(df) 
# # trying to fix inbalanced data
# from imblearn.under_sampling import RandomUnderSampler

# rus = RandomUnderSampler(random_state=42) 
# X_resampled, y_resampled = rus.fit_resample(df.drop("dep_delayed_15min", axis=1), df["dep_delayed_15min"])
# print(y_resampled.value_counts())

# # one hot encoding
# X_resampled = pd.get_dummies(X_resampled, drop_first=True) # drop_first=True kad nebutu multicollinearity
# # splitinsime data i train test ir validation
# X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,
#                                                      test_size=0.3, random_state=42) 
# # gauname validation set
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# # print(pd.Series(y_val).value_counts()) #
# # print(pd.Series(y_test).value_counts()) #

# # paprasciausias baseline modelis


# def model_predictions(model, X_train, y_train, X_val, y_val):
#     model.fit(X_train, y_train) # fit the model
#     y_pred = model.predict(X_val) # predict the model
#     print(classification_report(y_val, y_pred)) # classification report
#     print(pd.Series(y_pred).value_counts()) # kiek yra 0 ir 1

#     score = roc_auc_score(y_val, y_pred) # roc_auc_score 0.5 - atsiktinis speliojimas, 1 - tobulas modelis, 0 - blogas modelis
#     print(score)
#     return score

# from sklearn.preprocessing import StandardScaler
# # we don't want to scale binary columns, so we will drop them from the dataset so we can scale the rest of the columns
# scaler = StandardScaler()
# columns_to_scale = ['DepTime', 'Distance'] 
# X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
# X_val[columns_to_scale] = scaler.transform(X_val[columns_to_scale])

# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(max_iter=100) # max_iter=1000 kad butu daugiau iteraciju
# # model_predictions(model, X_train, y_train, X_val, y_val) # model predictions



# from sklearn.neighbors import KNeighborsClassifier

# # scores = []
# # kiekis = range(1,40)
# # for k in kiekis:
# #     model = KNeighborsClassifier(k) # max_iter=1000 kad butu daugiau iteraciju
# #     print('-'*80)
# #     print(k)
# #     scores.append(model_predictions(model, X_train, y_train, X_val, y_val)) # model predictions
# #     print('-'*80)

# # # plot scores with sns
# # sns.lineplot(x=kiekis, y=scores)
# # plt.title("KNN model scores")
# # plt.xlabel("K")
# # plt.ylabel("Score")
# # plt.show()

# # from sklearn.svm import SVC

# # # SVC modelis
# # model = SVC(kernel='linear')
# # model_predictions(model, X_train, y_train, X_val, y_val) # model predictions TOO SLOW


# df_test = pd.read_csv(r"data/flight_delays_test.csv")
# df_test = prepare_data(df_test) # prepare data
# df_test = pd.get_dummies(df_test, drop_first=True)
# df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])
# # before predicting we need to match the columns of the test set with the train set
# # we will drop the columns that are not in the train set
# # but since train might have columns that test doesn't have, we need to create those columns as null on the test set
# df_test = df_test.reindex(columns=X_train.columns, fill_value=0) # this will fill the missing columns with 0
# df_test = df_test[X_train.columns] # this will drop the columns that are not in the train set



# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# # # Random Forest modelis
# param_grid = {
#     'n_estimators': [10, 50],
#     'max_depth': [10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
# }
# # Grid search
# model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_estimators - kiek medziu
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')
# grid_search.fit(X_train, y_train)
# # Print best parameters
# print("Best parameters found: ", grid_search.best_params_)
# # Print best score
# print("Best score: ", grid_search.best_score_)

# model = grid_search.best_estimator_ # geriausias modelis
# model_predictions(model, X_train, y_train, X_val, y_val) # model predictions


# y_pred = model.predict(df_test) # predict the model

# # y_pred to csv with and and prediction for submission
# result_df = pd.DataFrame({
#     'id': df_test.index,
#     'dep_delayed_15min': y_pred
# })
# result_df.to_csv(r"submission.csv", index=False) # to csv without index