import pandas as pd
import numpy as np

# dictionary = {
#     'Name': [np.nan, np.nan, np.nan, 'Gabija', 'Simona', np.nan, np.nan, 'Lukas', 'Eglė', np.nan],
#     'Age': [30, 19, 45, 30, 32, 27, 22, 30, 29, 41],
#     'City': ['Vilnius', 'Kaunas', 'Klaipėda', 'Šiauliai', 'Panevėžys', 'Alytus', 'Marijampolė', 'Mažeikiai', 'Utena', 'Tauragė']
# }

# df = pd.DataFrame(dictionary)
# # print(df)

# # print(pd.DataFrame([[20,15,19],[5,10,7]], columns=['Testas','a','b']))

# # print(df.head(3))

# # print(df.shape)

# # df.info()
# # print(df.describe())
# # print(df['Age'].to_numpy().std(ddof=1))
# # print(df['Age'])
# # print(df[['Age','City']])
# # print(df.iloc[5])
# # print(
# #     df[df['Age'] > 30]
# #       )
# # df.rename(inplace=True,columns={'Age':'Amzius','Name':"Vardas"})
# # print(df)
# # print(df.isna())
# # print(df['Age'].isnull()) 
# # df.dropna(inplace=True)
# # print(df)

# # df['Naujas'] = range(0,10) # arba nurodyti viena reiksme kuri bus priskirta visom eilutem, arba nurodyti visas reiksmes 
# # print(df)
# # df.drop('Name',inplace=True,axis=1) # axis kontroliuoja ar kalbame apie eilute ar apie stulpeli
# # print(df)

# # df.replace(30,99,inplace=True)
# # print(df)
# # kint = df.groupby('Age')
# # print(kint.first())
# # dictionary = {
# #     'Name': [np.nan, np.nan, np.nan, 'Gabija', 'Simona', np.nan, np.nan, 'Lukas', 'Eglė', np.nan],
# #     'Age': [30, 19, 45, 30, 32, 27, 22, 30, 29, 41],
# #     'City': ['Vilnius', 'Kaunas', 'Klaipėda', 'Šiauliai', 'Panevėžys', 'Alytus', 'Marijampolė', 'Mažeikiai', 'Utena', 'Tauragė']
# # }
# # df2 = pd.DataFrame(dictionary)
# print(pd.concat([df,df],axis=1))
# print(pd.merge(df,df,on='Age',how='inner'))

# df = pd.read_csv(r'Titanic_train.csv')
# df = df.set_index('PassengerId')

# print(df)
# df.to_csv() # galime ir irasyti i faila

# print(df.transpose())

# import pandas as pd
# df = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/nba.csv")

# team = df.groupby('Team')
# for t in team.keys:
#     print(t) # Let's print the first entries in all the groups formed.
