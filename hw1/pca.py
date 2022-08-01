from os import listdir
import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train_set_name = [str(i)+"_"+str(j)+".png" for i in range(1,41) for j in range(1,7)]
test_set_name = [str(i)+"_"+str(j)+".png" for i in range(1,41) for j in range(7,11)]

train_X = [imageio.imread("hw1_dataset/"+n) for n in train_set_name]
train_X = np.array(train_X).reshape(240,-1)
train_y = [i for i in range(1,41) for _ in range(1,7)]

test_X = [imageio.imread("hw1_dataset/"+n) for n in test_set_name]
test_X = np.array(test_X).reshape(160,-1)
test_y = [i for i in range(1,41) for _ in range(7,11)]

#mean face
mean_vector = train_X.mean(axis=0)
'''
plt.title("mean face")
plt.imshow(mean_vector.reshape(56,46),cmap="gray")
plt.show()
'''

#first four eigenfaces
pca = PCA()
output = pca.fit(train_X-mean_vector)
output.components_.shape

e1 = (output.components_[0]).reshape(56,46)
e2 = (output.components_[1]).reshape(56,46)
e3 = (output.components_[2]).reshape(56,46)
e4 = (output.components_[3]).reshape(56,46)
'''
plt.title("eigenface1")
plt.imshow(e1,cmap="gray")
plt.show()
plt.title("eigenface2")
plt.imshow(e2,cmap="gray")
plt.show()
plt.title("eigenface3")
plt.imshow(e3,cmap="gray")
plt.show()
plt.title("eigenface4")
plt.imshow(e4,cmap="gray")
plt.show()
'''

first_img = imageio.imread("hw1_dataset/1_1.png").reshape(1,-1)
'''
plt.title("original")
plt.imshow(first_img.reshape(56,46), cmap="gray")
plt.show()
'''
projected =pca.transform(first_img-mean_vector)
#shape >> (1,240)


for j,i in enumerate([ 3, 45, 140, 229]):
    figure = (projected[:,:i] @ output.components_[:i]) + mean_vector
    mse = np.mean((figure - first_img)**2)
    plt.subplot(2,2,j+1)
    tit = "n="+str(i)+", mse="+str(np.round(mse,1))
    plt.title(tit)
    plt.imshow(figure.reshape(56,46), cmap = "gray")
#plt.show()

train_X_reduced = pca.transform(train_X - mean_vector)
'''
test the model
knn = KNeighborsClassifier()
grid = {"n_neighbors":[1,3,5]}
classify = GridSearchCV(knn, grid, cv=3) # conduct 3-fold cross validation
for n in [3, 45, 140]:
	classify.fit(train_X_reduced[:,:n], train_y)
	print("n= %3d" %n, classify.cv_results_["mean_test_score"])
'''
#true test
test_X_reduced = pca.transform(test_X - mean_vector)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_X_reduced[:,:45], train_y)
pred_y = knn.predict(test_X_reduced[:,:45])

acc = accuracy_score(y_pred=pred_y, y_true=test_y)
print("Accuracy:", acc)