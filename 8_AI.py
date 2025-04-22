# # from sklearn.cluster import KMeans
# # from sklearn.metrics import silhouette_score
# # from sklearn.datasets import make_blobs
# # import matplotlib.pyplot as plt
# # import seaborn as sns # for better visualization
# # import pandas as pd
# # import numpy as np


# # # Generate synthetic data
# # X, y = make_blobs(n_samples=300, centers=9, cluster_std=0.60, random_state=42)
# # print(X.shape)
# # # plot the data
# # plt.scatter(X[:, 0], X[:, 1], s=30) # s means size of the points
# # plt.title("Synthetic Data")
# # plt.xlabel("Feature 1")
# # plt.ylabel("Feature 2")
# # plt.show()

# # # elbow method to find the optimal number of clusters
# # wcss = [] # within cluster sum of squares
# # for i in range(1, 11):
# #     kmeans = KMeans(n_clusters=i, random_state=42, n_init=10, max_iter=300, verbose=0)
# #     kmeans.fit(X)
# #     wcss.append(kmeans.inertia_)

# # plt.plot(range(1, 11), wcss)
# # plt.title("Elbow Method")
# # plt.xlabel("Number of clusters")
# # plt.ylabel("WCSS")
# # plt.show()


# # # silhouette score to find the optimal number of clusters
# # silhouette_scores = [] # silhouette score
# # for i in range(2, 11):
# #     kmeans = KMeans(n_clusters=i, random_state=42, n_init=10, max_iter=300, verbose=0)
# #     kmeans.fit(X)
# #     silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# # # plot silhouette scores
# # plt.plot(range(2, 11), silhouette_scores)
# # plt.title("Silhouette Score")
# # plt.xlabel("Number of clusters")
# # plt.ylabel("Silhouette Score")
# # plt.show()
# # # best silhouette score is 0.9 for 4 clusters because it is the highest point on the graph


# # # Create a KMeans model
# # kmeans = KMeans(n_clusters=8, random_state=42, n_init=10, max_iter=300, verbose=0)
# # y_kmeans = kmeans.fit_predict(X) # fits the model and predicts the cluster for each data point


# # # plot the clusters
# # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='viridis') # X[row_start:row_end, column_start:column_end]
# # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100, alpha=0.5, marker='X')
# # plt.title("KMeans Clustering")
# # plt.xlabel("Feature 1")
# # plt.ylabel("Feature 2")
# # plt.show()

# # from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# # # Davies-Bouldin score
# # db_score = davies_bouldin_score(X, y_kmeans)
# # print("Davies-Bouldin score:", db_score) # stengiasi būti kuo mažesnis
# # # Calinski-Harabasz score
# # ch_score = calinski_harabasz_score(X, y_kmeans)
# # print("Calinski-Harabasz score:", ch_score) # stengiasi būti kuo didesnis


# # Hierarchical Clustering

# from scipy.cluster.hierarchy import dendrogram, linkage
# from sklearn.metrics import silhouette_score
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
# import seaborn as sns # for better visualization
# import pandas as pd
# import numpy as np

# # Generate synthetic data
# X, y = make_blobs(n_samples=300, centers=9, cluster_std=0.60, random_state=42)
# print(X.shape)
# # plot the data
# plt.scatter(X[:, 0], X[:, 1], s=30) # s means size of the points
# plt.title("Synthetic Data")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# # hierarchical clustering
# # Create the linkage matrix
# Z = linkage(X, method='ward')


# # plot the dendogram
# plt.figure(figsize=(10, 7))
# plt.title("Dendrogram")
# plt.xlabel("Data points")
# plt.ylabel("Distance")
# dendrogram(Z, leaf_rotation=90., leaf_font_size=12.)
# plt.show()

# # cut the dendrogram at a certain height to form clusters
# from scipy.cluster.hierarchy import fcluster
# from sklearn.metrics import davies_bouldin_score

# fclusters = fcluster(Z, t=20, criterion='maxclust') # t is the distance threshold. How to choose it? Choose the distance where the dendrogram has the largest vertical distance between two horizontal lines.
# print(fclusters) # shows the cluster for each data point

# # plot the clusters
# plt.scatter(X[:, 0], X[:, 1], c=fclusters, s=20, cmap='viridis') # X[row_start:row_end, column_start:column_end]
# plt.title("Hierarchical Clustering")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# # Davies-Bouldin score
# db_score = davies_bouldin_score(X, fclusters)
# print("Davies-Bouldin score:", db_score) # stengiasi būti kuo mažesnis



# •https://www.kaggle.com/datasets/shwetabh123/mall-customers
# •Parsisiųskite šiuos duomenis
# •Padarykite klasterizavimą (su dvejais skirtingais algoritmais)
# •Pateikite bent 3 išvadas ką galite pasakyti apie šiuos duomenis
# •Pasakykite kodėl jūsų manymų buvo atliktas tinkamas klasterizavimas ir kiek klasterių turėjo būti.

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import silhouette_score
# from sklearn.cluster import KMeans, DBSCAN
# from scipy.cluster.hierarchy import dendrogram, linkage
# import numpy as np
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# df = pd.read_csv(r'Mall_Customers.xls', index_col=0)
# # label encoder
# # from sklearn.preprocessing import LabelEncoder
# # le = LabelEncoder()
# # df['Genre'] = le.fit_transform(df['Genre'])

# df['IsMale'] = df['Genre'].apply(lambda x: 1 if x=='Male' else 0)
# df = df.drop('Genre',axis=1)
# # print(df.head())
# # df.info()

# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df.drop('IsMale', axis=1))

# df_scaled = pd.DataFrame(df_scaled, columns=df.columns[:-1], index=df.index)

# df_scaled['IsMale'] = df['IsMale']
# print(df_scaled.head())

# def elbow_method(df, max_clusters=10):
#     inertia = []
#     for i in range(1, max_clusters + 1):
#         kmeans = KMeans(n_clusters=i, random_state=42, init='random', n_init=100, max_iter=300)
#         kmeans.fit(df)
#         inertia.append(kmeans.inertia_)

#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, max_clusters + 1), inertia)
#     plt.title('Elbow Method')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Inertia')
#     plt.xticks(range(1, max_clusters + 1))
#     plt.grid()
#     plt.show()

# def silhouette_analysis(df, max_clusters=10):
#     sillhoutete_scores = []
#     for i in range(2, max_clusters + 1):
#         kmeans = KMeans(n_clusters=i, random_state=42, init='random', n_init=100, max_iter=300)
#         kmeans.fit(df)
#         sillhoutete_scores.append(silhouette_score(df, kmeans.labels_))
#     # plot the silhouette scores
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(2, max_clusters + 1), sillhoutete_scores)
#     plt.title('Silhouette Analysis')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Silhouette Score')
#     plt.xticks(range(2, max_clusters + 1))
#     plt.grid()
#     plt.show()




# def show_dendrogram(df, method='ward'):
#     Z = linkage(df, method=method)
#     plt.figure(figsize=(10, 7))
#     dendrogram(Z, labels=df.index, leaf_rotation=90)
#     plt.title(f'Dendrogram ({method})')
#     plt.xlabel('Customers')
#     plt.ylabel('Euclidean distances')
#     plt.show()

# def k_means_cluters(df, n_clusters):
#     # scaler = StandardScaler()
#     # df_scaled = scaler.fit_transform(df)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='random', n_init=100, max_iter=300) #arba 3
#     kmeans.fit(df)
#     return kmeans

# def dbscan_clusters(df, eps=0.5, min_samples=5):
#     scaler = StandardScaler()
#     df_scaled = scaler.fit_transform(df)
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     dbscan.fit_predict(df_scaled)
#     return dbscan

# def plot_clusters(df, opt):
#     fig, axes = plt.subplots(4, 3, figsize=(20, 5))
#     axes = axes.flatten()
#     ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = axes

#     ax1.scatter(df['Age'], df['Annual Income (k$)'], c=opt.labels_, cmap='viridis')
#     ax1.set_title('KMeans Clustering')
#     ax1.set_xlabel('Age')
#     ax1.set_ylabel('Annual Income (k$)')
#     ax2.scatter(df['Age'], df['Spending Score (1-100)'], c=opt.labels_, cmap='viridis')
#     ax2.set_title('KMeans Clustering')
#     ax2.set_xlabel('Age')
#     ax2.set_ylabel('Spending Score (1-100)')
#     ax3.scatter(df['Age'], df['IsMale'], c=opt.labels_, cmap='viridis')
#     ax3.set_title('KMeans Clustering')
#     ax3.set_xlabel('Age')
#     ax3.set_ylabel('IsMale')

#     ax4.scatter(df['Spending Score (1-100)'], df['Age'], c=opt.labels_, cmap='viridis')
#     ax4.set_title('KMeans Clustering')
#     ax4.set_xlabel('Spending Score (1-100)')
#     ax4.set_ylabel('Age')
#     ax5.scatter(df['Spending Score (1-100)'], df['Annual Income (k$)'], c=opt.labels_, cmap='viridis')
#     ax5.set_title('KMeans Clustering')
#     ax5.set_xlabel('Spending Score (1-100)')
#     ax5.set_ylabel('Annual Income (k$)')
#     ax6.scatter(df['Spending Score (1-100)'], df['IsMale'], c=opt.labels_, cmap='viridis')
#     ax6.set_title('KMeans Clustering')
#     ax6.set_xlabel('Spending Score (1-100)')
#     ax6.set_ylabel('IsMale')

#     ax7.scatter(df['Annual Income (k$)'], df['Age'], c=opt.labels_, cmap='viridis')
#     ax7.set_title('KMeans Clustering')
#     ax7.set_xlabel('Annual Income (k$)')
#     ax7.set_ylabel('Age')
#     ax8.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=opt.labels_, cmap='viridis')
#     ax8.set_title('KMeans Clustering')
#     ax8.set_xlabel('Annual Income (k$)')
#     ax8.set_ylabel('Spending Score (1-100)')
#     ax9.scatter(df['Annual Income (k$)'], df['IsMale'], c=opt.labels_, cmap='viridis')
#     ax9.set_title('KMeans Clustering')
#     ax9.set_xlabel('Annual Income (k$)')
#     ax9.set_ylabel('IsMale')


#     ax10.scatter(df['IsMale'], df['Age'], c=opt.labels_, cmap='viridis')
#     ax10.set_title('KMeans Clustering')
#     ax10.set_xlabel('IsMale')
#     ax10.set_ylabel('Age')
#     ax11.scatter(df['IsMale'], df['Annual Income (k$)'], c=opt.labels_, cmap='viridis')
#     ax11.set_title('KMeans Clustering')
#     ax11.set_xlabel('IsMale')
#     ax11.set_ylabel('Annual Income (k$)')
#     ax12.scatter(df['IsMale'], df['Spending Score (1-100)'], c=opt.labels_, cmap='viridis')
#     ax12.set_title('KMeans Clustering')
#     ax12.set_xlabel('IsMale')
#     ax12.set_ylabel('Spending Score (1-100)')

#     plt.suptitle('KMeans Clustering')
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.subplots_adjust(hspace=0.5)
#     plt.show()
# # plot clusters 3d
# def plot_clusters_3d(df, opt):
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], c=opt.labels_, cmap='viridis')
#     ax.set_title('KMeans Clustering')
#     ax.set_xlabel('Age')
#     ax.set_ylabel('Annual Income (k$)')
#     ax.set_zlabel('Spending Score (1-100)')
#     plt.show()


# # elbow_method(df_scaled, 10)
# # silhouette_analysis(df_scaled, 10)
# show_dendrogram(df_scaled, method='ward')
# km = k_means_cluters(df_scaled, 5)
# plot_clusters(df, km)
# plot_clusters_3d(df, km)
# df['Cluster'] = km.labels_
# # split the data into clusters
# clusters = df.groupby('Cluster')
# dataframes = []
# for name, group in clusters:
#     dataframes.append(group)
#     # print each dataframe information
#     print(group.describe())


# # print(dataframes)
