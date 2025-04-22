# def funkcija(skaicius): # skaicius = 5
#     if skaicius < 4:
#         return skaicius ** 3 # return - nutraukia funkcija
#     else:
#         return skaicius**2 # 5**2 -> 25

# print(funkcija(5)) # funkcija() -> 2

# def funkcija2(kint): # kint(naujas) = 7
#     kint = 12 # funkcijoje sukurtas kintamasis, neegzistuoja uz jos ribu
#     print("Sveiki")
#     return 0

# kint = 7
# funkcija2(kint) # funkcija2() -> 0
# print(kint)

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# # Load Titanic dataset
# titanic = sns.load_dataset('titanic')
# print(titanic)
# # Remove rows with missing 'age' or 'sex' or 'survived' values
# titanic_clean = titanic.dropna(subset=['age', 'sex', 'survived'])

# plt.figure(figsize=(10, 6))
# sns.histplot(data=titanic_clean[titanic_clean['survived'] == 1], x='age', hue='sex', multiple='stack', bins=30, stat='count')
# plt.title('Survived Passengers by Age and Gender')
# plt.xlabel('Age')
# plt.ylabel('Survivor Count')

# plt.show()


# bins = np.arange(0, 81, 10)
# labels = [f"{i}-{i+9}" for i in bins[:-1]]
# titanic_clean['age_group'] = pd.cut(titanic_clean['age'], bins=bins, labels=labels, right=False)

# grouped = titanic_clean.groupby(['sex', 'age_group']).agg(
#     total=('survived', 'count'),
#     survivors=('survived', 'sum')
# )
# grouped['survival_rate'] = grouped['survivors'] / grouped['total'] * 100

# sns.lineplot(data=grouped, x='age_group', y='survival_rate', hue='sex', marker="o")
# plt.show()