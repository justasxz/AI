import pandas as pd
import numpy as np

# sr = pd.Series([10,30,20,40])
# print(sr)


# sr = pd.Series(data=[10,30,20,40],index=["Jonas","Antanas","Mantas","Karolis"])
# print(sr)

# # print(sr.head(2))
# # print(sr.tail(2))
# # print(sr.size)
# # print(sr.values)
# # print(sr.index)

# # print(sr['Jonas':'Mantas']) # jeigu tiesiog naudojame lauztus skliaustus, tai kreipiames i indeksa
# print(sr.iloc[3]) # pagal pozicija
# # print(sr[sr >= 30])

# sr = sr.add(4)
# print(sr)
# print(sr.std())

# sr = pd.Series([
#     10.5, 23.1, np.nan, 47.8, 12.0,
#     35.6, np.nan, 18.3, 29.9, 41.2,
#     56.7, 67.0, 33.3, np.nan, 22.4,
#     90.1, 15.8, np.nan, 77.7, 64.3
# ])

# print(sr)
# # sr.dropna(inplace=True) # inplace = True - keicia originala, kitu atveju grazina kopija
# # # print(sr)
# # print(sr.isnull().sum()) # [0,0,1,0,0,1,0,0,1,0,1]
# sr.fillna(sr.mean(), inplace=True) # kuo uzpildyti reikai atsizvelgti i logika ir konteksta
# print(sr)

# sr = pd.Series([
#     "Labas", "Katinas", "Meska", "Tigras","Arklys"
#     ])

# print(sr.str.contains('ab'))


sr = pd.Series([
    10, 23.1, np.nan, 47.8, 12.0,
    35.6, np.nan, 18.3, 29.9, 41.2,
    56.7, 67.0, 33.3, np.nan, 18.3,
    90.1, 15.8, np.nan, 77.7, 18.3
])

# print(sr)
# # print(sr.value_counts()) # kiekviena reiksme ir kiek jos yra
# # print(sr.unique()) # grazinas unikaliu reiksmiu sarasa
# # print(sr.nunique()) # grazina kiek yra unikaliu reiksmiu

# # sr = sr.apply(lambda x: x /2 *3) # apply pritaiko bet kokia funkcija visiems elementams
# print(sr)
# diction = {10:0,23.1:1}

# print(sr.map(diction)) 

# sr.sort_values(inplace=True)
# sr.reset_index(drop=True,inplace=True) # Nustato indeksa is naujo ir pakeicia
# print(sr)

