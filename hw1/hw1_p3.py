import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from mpl_toolkits.mplot3d import Axes3D

IMAGES_PATH = './p3_data/'
PATCH_SIZE = 16
TRAIN_SIZE = 375
TEST_SIZE = 125
CLASS_NAMES = ['banana', 'fountain', 'reef', 'tractor']
CLASS_NUM = len(CLASS_NAMES)
CLUSTER_NUM = 15

def q1():
    image_path_dict = { class_name: os.listdir(os.path.join(IMAGES_PATH, class_name)) for class_name in CLASS_NAMES }
   
    train_patches = []
    test_patches = []
    chosen_patches = []
    chosen_images = []
    for i, image_pathes in enumerate(image_path_dict.items()):
        class_name = image_pathes[0]
        image_pathes = image_pathes[1]
        image_pathes = [os.path.join(IMAGES_PATH, class_name, image_path) for image_path in image_pathes]
        for j, image_path in enumerate(image_pathes):
            img = cv2.imread(image_path)
            for w in range(0,img.shape[0],PATCH_SIZE):
                for h in range(0,img.shape[1],PATCH_SIZE):
                    patch = img[w:w+PATCH_SIZE, h:h+PATCH_SIZE,:]
                    if j < TRAIN_SIZE:
                        train_patches.append(patch)
                    else:
                        test_patches.append(patch)
        patch_choice = random.randint(0, TRAIN_SIZE)
        chosen_images.append(cv2.imread(image_pathes[patch_choice]))
        patch_choices = [ random.randint(0, 4*4) + 4*4*patch_choice for _ in range(3) ]
        chosen_patches.append([train_patches[i*TRAIN_SIZE*4*4+choice] for choice in patch_choices])

    vcat = None
    for i, patches in enumerate(chosen_patches):
        patches = map(lambda x:cv2.copyMakeBorder(x,2,2,2,2,cv2.BORDER_CONSTANT,value=[0,0,0]), patches)
        hcat = cv2.hconcat([patch for patch in patches])
        if i == 0:
            vcat = hcat
        else:
            vcat = cv2.vconcat([vcat,hcat])
    cv2.imwrite("p3_q1.png", vcat)

    hcat = cv2.hconcat(chosen_images)
    cv2.imwrite("p3_q1_images.png", hcat)
        
    train_patches = np.array(train_patches)
    test_patches = np.array(test_patches)
    X_train_patches = train_patches.reshape(train_patches.shape[0], -1)
    X_test_patches = test_patches.reshape(test_patches.shape[0], -1)
    
    return X_train_patches, X_test_patches

def q2(X_train_patches, X_test_patches):
    # print(X_train_patches.shape, X_test_patches.shape)
    km = KMeans(n_clusters=CLUSTER_NUM, max_iter=5000, n_jobs=-1).fit(X_train_patches)
    centers = km.cluster_centers_
    # labels are the indexes of centers
    labels = km.labels_
    pca = PCA(n_components=3).fit(X_train_patches)
    train_patches_pca = pca.transform(X_train_patches)
    centers_pca = pca.transform(centers)

    center_indexes = [ i for i in range(centers_pca.shape[0]) ]
    center_choices = random.sample(center_indexes, 6)
    # print("Chosen clusters' index: {}".format(center_choices))
    plot_data = list()
    for center_index in center_choices:
        for index, label in enumerate(labels):
            if label == center_index:
                plot_data.append(np.append(train_patches_pca[index], center_index)) 

    plot_data = np.array(plot_data)
    plot_centers = np.array([ np.append(centers_pca[choice], choice) for choice in center_choices ])
    fig = plt.figure()
    ax = Axes3D(fig)
    # print(plot_data.shape, plot_centers)
    ax.scatter(plot_data[:,0], plot_data[:,1], plot_data[:,2], c=plot_data[:,3], s=1, cmap='plasma', alpha=0.2)
    s = ax.scatter(plot_centers[:,0], plot_centers[:,1], plot_centers[:,2], c='black', s=200, marker='D')
    s.set_edgecolors = s.set_facecolors = lambda *args:None
    fig.savefig('p3_q2.png')
    return km, pca

def BoW(patches, centers):
    softmax = np.zeros(CLUSTER_NUM)
    for patch in patches:
        distance = [ np.linalg.norm(patch-center,ord=2) for center in centers ]
        distance = np.asarray(distance)
        recip = np.reciprocal(distance)
        recip = recip / np.sum(recip)
        softmax = [softmax[center_index] if softmax[center_index]>recip[center_index] else recip[center_index] for center_index in range(CLUSTER_NUM)]
    return softmax

def q3(X_train_patches, X_test_patches, km):
    
    image_choices = [ random.randint(0, TRAIN_SIZE)*4*4 for _ in range(CLASS_NUM) ]
    centers = km.cluster_centers_
    for i, choice in enumerate(image_choices):
        chosen_patches = X_train_patches[choice:choice+4*4]
        softmax = BoW(chosen_patches, centers) 
        plt.figure()
        index = np.arange(1,CLUSTER_NUM+1)
        plt.bar(index, softmax)
        plt.savefig('p3_q4_{}.png'.format(CLASS_NAMES[i]))

def q4(X_train_patches, X_test_patches, km):
    centers = km.cluster_centers_
    bow_train_data = list()
    bow_test_data = list()
    for i in range(0,CLASS_NUM*TRAIN_SIZE):
        # Each patches represent one image
        patches = X_train_patches[i*4*4 : i*4*4 + 4*4]
        bow_train_data.append(BoW(patches, centers))

    for i in range(0,CLASS_NUM*TEST_SIZE):
        patches = X_test_patches[i*4*4 : i*4*4 + 4*4]
        bow_test_data.append(BoW(patches, centers))

    bow_train_data = np.asarray(bow_train_data)
    bow_test_data = np.asarray(bow_test_data)
    train_labels = [[i]*TRAIN_SIZE for i in range(CLASS_NUM)]
    train_labels = [ item for sublist in train_labels for item in sublist ]
    test_labels = [[i]*TEST_SIZE for i in range(CLASS_NUM)]
    test_labels = [ item for sublist in test_labels for item in sublist ]
    knn = KNN(n_neighbors=5).fit(bow_train_data, train_labels)
    acc = knn.score(bow_test_data, test_labels)
    print("k-nearest neighbors classifier accuracy: {}".format(acc))   

if __name__ == '__main__':
    X_train_patches, X_test_patches = q1()  
    km, pca = q2(X_train_patches, X_test_patches)
    q3(X_train_patches, X_test_patches, km)
    q4(X_train_patches, X_test_patches, km)
    


