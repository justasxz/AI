# # import numpy as np
# # import matplotlib.pyplot as plt
# # from skimage import data, color
# # from skimage.transform import resize
# # from skimage.feature import local_binary_pattern, hog
# # import cv2

# # # # ------------------------------------------------------------------------
# # # # Feature Extraction Script with Visualizations
# # # # ------------------------------------------------------------------------
# # # # Sections:
# # # # 1. Preprocessing
# # # # 2. Color Histogram
# # # # 3. Local Binary Patterns (LBP)
# # # # 4. Histogram of Oriented Gradients (HOG)
# # # # 5. ORB Keypoints
# # # # 6. Classification Pipeline (commented out)

# # # 1. Preprocessing
# # def preprocess_rgb(img, size=(64, 64)):
# #     """
# #     Resize RGB image to 'size', convert to grayscale, normalize to [0,1].
# #     Returns: (resized RGB, grayscale, normalized grayscale)
# #     """
# #     img_resized = resize(img, size, anti_aliasing=True)
# #     img_gray = color.rgb2gray(img_resized)
# #     img_norm = img_gray / np.max(img_gray)
# #     return img_resized, img_gray, img_norm

# # # # Load sample RGB images
# # img_astronaut = data.astronaut()
# # img_coffee = data.coffee()

# # ast_rgb, ast_gray, ast_norm = preprocess_rgb(img_astronaut)
# # cf_rgb, cf_gray, cf_norm   = preprocess_rgb(img_coffee)

# # # Visualize preprocessing steps for astronaut
# # plt.figure()
# # plt.subplot(1, 4, 1)
# # plt.imshow(cf_rgb)
# # plt.title('Resized RGB')
# # plt.axis('off')

# # plt.subplot(1, 4, 2)
# # plt.imshow(cf_gray, cmap='gray')
# # plt.title('Grayscale')
# # plt.axis('off')

# # plt.subplot(1, 4, 3)
# # plt.imshow(cf_norm, cmap='gray')
# # plt.title('Normalized Grayscale')
# # plt.axis('off')

# # plt.figure()
# # plt.subplot(1, 1, 1)
# # plt.imshow(img_coffee)
# # plt.title('Resized RGB')
# # plt.axis('off')
# # # plt.show()

# # # # 2. Color Histogram
# # def color_histogram(img_rgb, bins=16):
# #     """
# #     Compute concatenated histograms for each RGB channel.
# #     Returns a 3*bins-length feature vector.
# #     """
# #     hists = []
# #     for ch in range(3):
# #         hist, _ = np.histogram(img_rgb[:, :, ch], bins=bins, range=(0, 1), density=True)
# #         hists.append(hist)
# #     return np.concatenate(hists)

# # # Visualize color histogram for coffee image
# # hist_cf_color = color_histogram(cf_rgb)
# # plt.figure()
# # plt.plot(hist_cf_color)
# # plt.title('Color Histogram (Coffee)')
# # plt.xlabel('Bin index')
# # plt.ylabel('Density')
# # # plt.show()

# # # # 3. Local Binary Patterns (LBP)
# # def lbp_features(img_gray, P=8, R=1, bins=16):
# #     """
# #     Compute histogram of Local Binary Pattern codes.
# #     """
# #     lbp = local_binary_pattern(img_gray, P, R, method='uniform')
# #     hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, P + 2), density=True)
# #     return lbp, hist

# # # Compute and visualize LBP for astronaut
# # lbp_ast, hist_ast_lbp = lbp_features(cf_norm)
# # plt.figure()
# # plt.subplot(1, 2, 1)
# # plt.imshow(lbp_ast, cmap='gray')
# # plt.title('LBP Image (Astronaut)')
# # plt.axis('off')

# # plt.subplot(1, 2, 2)
# # plt.bar(range(len(hist_ast_lbp)), hist_ast_lbp)
# # plt.title('LBP Histogram')
# # plt.xlabel('LBP Code')
# # plt.ylabel('Density')
# # plt.show()

# # # # 4. Histogram of Oriented Gradients (HOG)
# # def hog_features(img_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
# #     """
# #     Compute HOG descriptor and visualization image.
# #     """
# #     hog_vec, hog_img = hog(
# #         img_gray,
# #         orientations=9,
# #         pixels_per_cell=pixels_per_cell,
# #         cells_per_block=cells_per_block,
# #         block_norm='L2-Hys',
# #         visualize=True
# #     )
# #     return hog_vec, hog_img

# # # Compute and visualize HOG for coffee
# # hog_cf_vec, hog_cf_img = hog_features(cf_gray)
# # plt.figure()
# # plt.imshow(hog_cf_img, cmap='gray')
# # plt.title('HOG Visualization (Coffee)')
# # plt.axis('off')
# # # plt.show()

# # # 5. ORB Keypoints
# # def orb_features(img_gray, n_keypoints=100):
# #     """
# #     Detect ORB keypoints and return the keypoints-overlaid image.
# #     """
# #     orb = cv2.ORB_create(nfeatures=n_keypoints)
# #     kp, des = orb.detectAndCompute((img_gray * 255).astype('uint8'), None)
# #     img_kp = cv2.drawKeypoints((img_gray * 255).astype('uint8'), kp, None, flags=0)
# #     # Convert BGR to RGB if needed (here grayscale so it's fine)
# #     return img_kp, des

# # # Compute and visualize ORB for astronaut
# # orb_ast_img, orb_ast_desc = orb_features(ast_norm)
# # plt.figure()
# # plt.imshow(orb_ast_img, cmap='gray')
# # plt.title('ORB Keypoints (Astronaut)')
# # plt.axis('off')
# # plt.show()

# # # 6. Classification Pipeline (commented out)

# from sklearn import datasets
# from skimage.feature import hog
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
# import pandas as pd
# import numpy as np
# # Load digits dataset
# digits = datasets.load_digits()
# # print(pd.DataFrame(digits.images).head())  # Print the first few rows of the dataset
# X_digits = digits.images
# y = digits.target # Print the shape of the images array

# # Extract HOG features for each digit
# X_hog = np.array([
#     hog(img, orientations=9, pixels_per_cell=(4, 4),
#         cells_per_block=(1, 1), block_norm='L2-Hys')
#     for img in X_digits
# ])


# print(X_hog.shape)  # Print the shape of the HOG features array

# # # Train/test split
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X_hog, y, test_size=0.2, random_state=42, stratify=y
# # )

# # # Define classifiers
# # classifiers = {
# #     'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=0),
# #     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
# #     'SVM (RBF)':      SVC(kernel='rbf', C=10, gamma='scale'),
# #     'k-NN (k=5)':     KNeighborsClassifier(n_neighbors=5)
# # }

# # # # Train & evaluate
# # for name, clf in classifiers.items():
# #     clf.fit(X_train, y_train)
# #     preds = clf.predict(X_test)
# #     print(f"=== {name} ===")
# #     print(classification_report(y_test, preds))


# # import cv2
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # image = cv2.imread('resized-image.jpeg', cv2.IMREAD_GRAYSCALE)
# # plt.imshow(image, cmap='gray')
# # plt.axis('off')
# # plt.show()

# # array = np.array(image)
# # # print(array)  # Print the array representation of the image
# # # print(array.shape)  # Print the shape of the array
# # # print(array.reshape(-1, 3).shape)  # Reshape the array to 1 row and 3 columns
# # print(pd.DataFrame(array.reshape(1, 4096)))  # Convert to DataFrame and print


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# df = pd.read_csv('sign_mnist_train.csv')

# print(df.head())  # Print the first few rows of the DataFrame

# # # show image with imshow
# # plt.imshow(df.iloc[0, 1:].values.reshape(28, 28), cmap='gray')
# # plt.axis('off')
# # plt.title(f"Label: {df.iloc[0, 0]}")
# # plt.show()
# plt.imshow(np.random.rand(28, 28), cmap='gray')
# plt.axis('off')
# plt.title(f"Label: {np.random.randint(0, 25)}")
# plt.show()


