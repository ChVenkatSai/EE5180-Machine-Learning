#!/usr/bin/env python
# coding: utf-8

# # Problem 1: K means clustering
# 

# #### Perform k means clustering on the [r,g,b] pixel data of the image 'chilis.jpg' for k = 3 clusters. Terminate the algorithm when the cluster means do not change in an iteration. 
# 
# You can initialize the cluster centers as:
# *   $c_1=[255, 0, 0]$
# *   $c_2=[0,0,0]$
# *   $c_3=[255,255,255]$
# 
# In order to visualise the output, replace all pixels corresponding to a cluster with it's mean value. Display this image along with the cluster means.
# 
# Code the algorithm from scratch without using libraries like scikit-learn.  
# 
# 
# 

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import imageio  

def Comp_arrays(A,B):
    for i in range (0,3):
        for j in range(0,3):
            if((A[i][j]==B[i][j])==0):
                
                return 0 
    return 1

def read_image():
      
    # loading the png image as a 3d matrix 
    img = imageio.imread('chilis.jpg') 
  
    # uncomment the below code to view the loaded image
    # plt.imshow(A) # plotting the image
    # plt.show() 
      
    # scaling it so that the values are small
    img = img / 255 
  
    return img
  
def initialize_means(img, clusters):
      
    # reshaping it or flattening it into a 2d matrix
    points = np.reshape(img, (img.shape[0] * img.shape[1],img.shape[2])) 
    m, n = points.shape
  
    # clusters is the number of clusters
    # or the number of colors that we choose.
      
    # means is the array of assumed means or centroids. 
    means = np.array([[1-1e-6,0,0],[0,0,0],[1-1e-6,1-1e-6,1-1e-6]]) 
  
    return points, means
  
  
# Function to measure the euclidean
# distance (distance formula)
def distance(x1, y1, x2, y2):
      
    dist = np.square(x1 - x2) + np.square(y1 - y2)
    dist = np.sqrt(dist)                                            # Euclidean Distance
    return dist
  
def k_means(points, means, clusters):
  
    iterations = 10 # the number of iterations 
    m, n = points.shape
      
    # these are the index values that
    # correspond to the cluster to
    # which each pixel belongs to.
    index = np.zeros(m) 
  
    # k-means algorithm.
    while(iterations > 0):
        for j in range(len(points)):
            # initialize minimum value to a large value
            minv = 1000
            temp = None
              
            for k in range(clusters):
                x1 = points[j, 0]
                y1 = points[j, 1]
                x2 = means[k, 0]
                y2 = means[k, 1]
              
                if(distance(x1, y1, x2, y2) < minv):         # checking distance of a point from the 3 centroids
                    minv = distance(x1, y1, x2, y2)             
                    temp = k
                    index[j] = k 
          
        for k in range(clusters):
            sumx = 0
            sumy = 0
            count = 0
              
            for j in range(len(points)):
                  
                if(index[j] == k):
                    sumx += points[j, 0]
                    sumy += points[j, 1] 
                    count += 1
              
            if(count == 0):
                count = 1    
              
            means[k, 0] = float(sumx / count)                 # averaging and getting new centroid
            means[k, 1] = float(sumy / count)                 # averaging and getting new centroid
                  
        iterations -= 1
  
    return means, index
  
  
def compress_image(means, index, img):
  
    # recovering the compressed image by
    # assigning each pixel to its corresponding centroid.
    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]
      
    # getting back the 3d matrix (row, col, rgb(3))
    recovered = np.reshape(recovered, (img.shape[0], img.shape[1],img.shape[2]))      # reshaping into image
  
    # plotting the compressed image.
    plt.imshow(recovered)
    plt.show()
  
    # saving the compressed image.
    imageio.imsave('compressed_' + str(clusters) +'_colors.png', recovered)
  
img = read_image()
  
clusters = 3
#clusters = int(input('Enter the number of colors in the compressed image. default = 16\n'))##
points, means = initialize_means(img, clusters)
means, index = k_means(points, means, clusters)
compress_image(means, index, img)
print("Final Centroids:\n",means*255)


# #Problem 2: S.V.M

# The Support Vector Machine(S.V.M) algorithm is to find the hyperplane in the N-dimensional space (N-Number of features) that distinctly classifies the data points.
# 
# To separate the two classes of data points, there are many possible hyperplanes that could be chosen. The objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

# ##Problem 2, Part A: Linear S.V.M
# 

# Use dataset A (``data_prob2_parta.csv``)  for this part of the question. The given CSV file has three columns: column 1 is the first input feature, column 2 is the second input feature and column 3 is the output label. Split the dataset into the training data (75%) and testing data(25%) randomly.
# 
# 
# 

# In[32]:


import pandas as pd
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

data=pd.read_csv("data_prob2_parta.csv").values
data=pd.DataFrame(data = data,columns = ['Input feature 1','Input feature 2','Output label'])
X = data[['Input feature 1', 'Input feature 2']]
y = data['Output label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)    # Spilting data into 75% and 25%


# Visualize the training data with a scatter plot (input feature 1 on the X axis, input feature 2 on the Y axis and color the points according to their labels)

# In[33]:


X_train0=X_train[y_train==0].values
X_train1=X_train[y_train==1].values
X_test0=X_test[y_test==0].values
X_test1=X_test[y_test==1].values

figure(1)
scatter(X_train0[:,0],X_train0[:,1],s=5,c="blue",label='0')
scatter(X_train1[:,0],X_train1[:,1],s=5,c="red",label='1')
xlabel('input feature 1')
ylabel('input feature 2')
legend(bbox_to_anchor = (1.2, 0.6))
show()


# Build the Support Vector Machine  model using the 
# training data. The scikit
# library can be used to build the model.

# In[34]:


svm=SVC(kernel='linear')
model=svm.fit(X_train,y_train)                           # fitting using SVM model
w1,w2=model.coef_.T                                      # getting coefficients or weights
b=model.intercept_[0]
SV=model.support_vectors_


# Print the parameter and support vectors.
# 

# In[35]:


print("weights:","w1=",w1[0],"and w2=",w2[0])
print("intercept:",b)
print("Support Vectors:\n",SV)


# Print the final accuracy on the test data.
# 

# In[36]:


print('Accuracy: ',model.score(X_test,y_test))


# Plot the scatter plot for the test data. On top of this scatter plot, plot the separating hyperplane and parallels to the hyperplane that pass through the support vectors.
# 
# 
# 
# 

# In[37]:


xx = np.linspace(data.values[:,0].min()-1,data.values[:,0].max()+1,500)
yy = np.linspace(data.values[:,1].min()-1,data.values[:,1].max()+1,500)
YY, XX = meshgrid(yy, xx)
xy = vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)
a=-(w1[0]/w2[0])
y=a*xx-(b/w2[0])

for i in range(len(SV)):
  ys=a*xx+(SV[i][1]-a*SV[i][0])
  plot(xx,ys,'g--')
plot(xx,y)
# plot decision boundary and margins
#contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
# plot support vectors
scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=10,
           linewidth=1, facecolors='none', edgecolors='k')
scatter(X_test0[:,0],X_test0[:,1],s=5,c="blue",label='0')
scatter(X_test1[:,0],X_test1[:,1],s=5,c="red",label='1')


# ##Problem 2, Part B: Non-linear S.V.M

#  Use Dataset B (``data_prob2_partb.csv``) for this part of the question. The given CSV file has three columns: column 1 is the first input feature, column 2 is the second input feature and column 3 is the output label. Split the dataset into training data (75%) and testing data (25%) randomly.
# 

# In[38]:


import pandas as pd
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

data=pd.read_csv("data_prob2_partb.csv").values
data=pd.DataFrame(data = data,columns = ['Input feature 1','Input feature 2','Output label'])
X = data[['Input feature 1', 'Input feature 2']]
y = data['Output label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)    # Spilting data into 75% and 25%


# Visualize the training data with a scatter plot (input feature 1 on the X axis, input feature 2 on the Y axis and color the points according to their labels).
# 

# In[39]:


X_train0=X_train[y_train==0].values
X_train1=X_train[y_train==1].values
X_test0=X_test[y_test==0].values
X_test1=X_test[y_test==1].values

figure(1)
scatter(X_train0[:,0],X_train0[:,1],s=5,c="blue",label='0')
scatter(X_train1[:,0],X_train1[:,1],s=5,c="red",label='1')
xlabel('input feature 1')
ylabel('input feature 2')
legend(bbox_to_anchor = (1.2, 0.6))
show()


# 
# 
# Write the code for
# choosing best hyperparameters for each of the kernel type.
# In text cell after that report the following numbers:
# Report the best kernel parameters, regularisation parametes, accuracy for ***linear***, ***RBF*** and ***poly*** kernels(Linear kernels has no kernel parameter).
# 
# Note: The scikit library can be used in this case.

# In[310]:


# using linear kernel
C_param={'C':[v for v in linspace(1e-4,100,10)]}
svmlinear=SVC(kernel='linear')
clf_lin = GridSearchCV(svmlinear, C_param)
clf_lin.fit(X_train,y_train)
print(clf_lin.best_estimator_)

# using rbf kernel

C_param={'C':[v for v in linspace(11,50,100)],'gamma':[v for v in linspace(1e-3,1,20)]}
svmRBF=SVC(kernel='rbf')
clf_rbf = GridSearchCV(svmRBF, C_param)
clf_rbf.fit(X_train,y_train)
print(clf_rbf.best_estimator_)

# using poly kernel

C_param={'C':[v for v in linspace(1e-3,1,10)],'degree':[v for v in linspace(1,10,10)]}
svmpoly=SVC(kernel='poly')
clf_poly = GridSearchCV(svmpoly, C_param)
clf_poly.fit(X_train,y_train)
print(clf_poly.best_estimator_)


# In[ ]:


model_lin=SVC(C=0.0001,kernel='linear').fit(X_train,y_train)
print('Best CV score linear kernel:',clf_lin.best_score_)
print('Accuracy_lin: ',model_lin.score(X_test,y_test))

model_rbf=SVC(C=26.363,gamma=0.053,kernel='rbf').fit(X_train,y_train)
print('Best CV score rbf kernel:',clf_rbf.best_score_)
print('Accuracy_rbf: ',model_rbf.score(X_test,y_test))

model_poly=SVC(C=0.334,degree=4,kernel='poly').fit(X_train,y_train)
print('Best CV score poly kernel:',clf_poly.best_score_)
print('Accuracy_poly: ',model_poly.score(X_test,y_test))


# Report your observation in the given table:
# 
# 
# 
# 
# 
# Kernels | Linear | RBF | Poly
# --- | --- | --- |---
# Kernel Parameters |None  | 0.053 | 4
# Regularization Parameters |0.0001|26.363 | 0.334
# Accuracy |0.497|0.979|0.979
# 
# 
# 
# 

# Plot the scatter plot for the test data.On top of this scatter plot, plot the decision regions for each of the kernels with their best fit

# In[49]:


svm_lin=SVC(C=0.0001,kernel='linear')
model_lin=svm_lin.fit(X_train,y_train)
print('Accuracy linear: ',model_lin.score(X_test,y_test))

svm_rbf=SVC(C=26.363,gamma=0.053,kernel='rbf')
model_rbf=svm_rbf.fit(X_train,y_train)
print('Accuracy rbf: ',model_rbf.score(X_test,y_test))

svm_poly=SVC(C=0.334,degree=4,kernel='poly')
model_poly=svm_poly.fit(X_train,y_train)
print('Accuracy poly: ',model_poly.score(X_test,y_test))

xx = np.linspace(data.values[:,0].min()-1,data.values[:,0].max()+1,500)
yy = np.linspace(data.values[:,1].min()-1,data.values[:,1].max()+1,500)
YY, XX = meshgrid(yy, xx)
Z_lin = model_lin.predict(c_[XX.ravel(), YY.ravel()])
Z_lin = Z_lin.reshape(XX.shape)
Z_rbf = model_rbf.predict(c_[XX.ravel(), YY.ravel()])
Z_rbf = Z_rbf.reshape(XX.shape)
Z_poly = model_poly.predict(c_[XX.ravel(), YY.ravel()])
Z_poly = Z_poly.reshape(XX.shape)

figure()
# plot decision boundary
contour(XX, YY, Z_lin,cmap=cm.Spectral, alpha=0.8)
# scatter plot of test data
scatter(X_test0[:,0],X_test0[:,1],s=5,c="blue",label='0')
scatter(X_test1[:,0],X_test1[:,1],s=5,c="red",label='1')
legend(bbox_to_anchor = (1.2, 0.6))

figure()
# plot decision boundary
contour(XX, YY, Z_rbf,cmap=cm.Spectral, alpha=0.8)
# scatter plot of test data
scatter(X_test0[:,0],X_test0[:,1],s=5,c="blue",label='0')
scatter(X_test1[:,0],X_test1[:,1],s=5,c="red",label='1')
legend(bbox_to_anchor = (1.2, 0.6))

figure()
# plot decision boundary
contour(XX, YY, Z_poly,cmap=cm.coolwarm, alpha=0.8)
# scatter plot of test data
scatter(X_test0[:,0],X_test0[:,1],s=5,c="blue",label='0')
scatter(X_test1[:,0],X_test1[:,1],s=5,c="red",label='1')
legend(bbox_to_anchor = (1.2, 0.6))


# ##Problem 2, Part C: Multiclass Classification

# Use Dataset C (``data_prob3_partc.csv``) for this part of the question. The given CSV file has three columns: column 1 is the first input feature, column 2 is the second input feature and column 3 is the output label. Split the dataset into training data (75%) and testing data (25%) randomly.
# 

# In[50]:


import pandas as pd
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

data=pd.read_csv("data_prob2_partc.csv").values
data=pd.DataFrame(data = data,columns = ['Input feature 1','Input feature 2','Output label'])
X = data[['Input feature 1', 'Input feature 2']]
y = data['Output label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)    # Spilting data into 75% and 25%


# Visualize the training data with a scatter plot (input feature 1 on the X axis, input feature 2 on the Y axis and color the points according to their labels).

# In[51]:


X_train0=X_train[y_train==0].values
X_train1=X_train[y_train==1].values
X_train2=X_train[y_train==2].values
X_test0=X_test[y_test==0].values
X_test1=X_test[y_test==1].values
X_test2=X_test[y_test==2].values
figure(3)
scatter(X_train0[:,0],X_train0[:,1],s=5,c="blue",label='0')
scatter(X_train1[:,0],X_train1[:,1],s=5,c="red",label='1')
scatter(X_train2[:,0],X_train2[:,1],s=5,c="green",label='2')
xlabel('input feature 1')
ylabel('input feature 2')
legend(bbox_to_anchor = (1.2, 0.6))
show()


# In[ ]:


C_param={'C':[v for v in linspace(90,100,100)],'gamma':[v for v in linspace(1e-3,1,20)]}
svmRBF=SVC(kernel='rbf')
clf_rbf = GridSearchCV(svmRBF, C_param)
clf_rbf.fit(X_train,y_train)
print(clf_rbf.best_estimator_)


# Build the Support Vector Machine  model using the 
# training data. The scikit
# library can be used to build the model.

# In[53]:


svm=SVC(C=97.57,gamma=0.263,kernel='rbf',decision_function_shape='ovr')
model=svm.fit(X_train,y_train)                           # fitting using SVM model


# Print the final accuracy on the test data.

# In[54]:


print('Accuracy: ',model.score(X_test,y_test))


# Plot the scatter plot for the test data. On top of this scatter plot, plot the decision boundary.

# In[55]:


xx = np.linspace(data.values[:,0].min()-1,data.values[:,0].max()+1,500)
yy = np.linspace(data.values[:,1].min()-1,data.values[:,1].max()+1,500)
YY, XX = meshgrid(yy, xx)
Z = model.predict(c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)

# plot decision boundary
contour(XX, YY, Z,cmap=cm.coolwarm_r, alpha=0.8)
# scatter plot of test data
#scatter(X_test.values[:, 0], X_test.values[:, 1],s=5, c=y_test, cmap=plt.cm.Spectral)
scatter(X_test0[:,0],X_test0[:,1],s=5,c="blue",label='0')
scatter(X_test1[:,0],X_test1[:,1],s=5,c="red",label='1')
scatter(X_test2[:,0],X_test2[:,1],s=5,c="green",label='2')
legend(bbox_to_anchor = (1.2, 0.6))


# # Problem 3 : Principal Component Analysis

# #### In this exercise you will perform face recognition using eigenfaces. Face recognition can be formulated as a classification task, where the inputs are images and the outputs are people's names.

# Load grayscale images from the LFW - Labeled faces in the Wild dataset using scikit-learn. To reduce the number of classes, retain pictures of only those people that have atleast 100 different pictures. (already done for you here) </br></br>
# Report the number of images and the size of each image.

# In[15]:


from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

lfw_people = fetch_lfw_people(min_faces_per_person=100)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# Split the dataset into training and test sets in the ratio - 7:3.

# In[16]:


# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Use scikit-learn's PCA class to perform dimensionality reduction on the training set images. Reduce the dimensions to 100 principal components. These principal components are the eigenfaces.

# In[17]:


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
def extract_pc(n_comp,X_train,X_test,h,w):
  print("Extracting the top %d eigenfaces from %d faces" % (n_comp, X_train.shape[0]))
  pca = PCA(n_components=n_comp, svd_solver='randomized',whiten=True)
  X_pca=pca.fit(X_train)
  eigenfaces = X_pca.components_.reshape((n_comp, h, w))
  print("Variance:",sum(var(eigenfaces,axis=0)))
  X_train_pca = X_pca.transform(X_train)           # Projecting the input data on the eigenfaces orthonormal basis
  X_test_pca = X_pca.transform(X_test)
  return pca,X_pca,eigenfaces,X_train_pca,X_test_pca

pca,X_pca,eigenfaces,X_train_pca,X_test_pca=extract_pc(100,X_train,X_test,h,w)


# Reshape the principal eigenvectors into images and visualize the eigenfaces. Display 10 eigenfaces. 

# In[20]:


def plot_gallery(images,titles,h, w, n_row=2, n_col=5):
    """Helper function to plot a gallery of portraits"""
    
    figure(figsize=(1.8 * n_col, 2.4 * n_row))
    subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        subplot(n_row, n_col, i + 1)
        imshow(images[i].reshape((h, w)),cmap=cm.bone)
        title(titles[i], size=12)
        xticks(())
        yticks(())
def plot_eigenfaces(n_comp,eigenfaces,h,w):
  print("10 eigenfaces in %d eigenfaces" % n_comp)
  eigenface_titles = ["eigenface %d" % (i+1) for i in range(eigenfaces.shape[0])]
  plot_gallery(eigenfaces,eigenface_titles,h, w)

plot_eigenfaces(100,eigenfaces,h,w)


# Reconstruct any image (from training / test set) by projecting the image onto the new eigenface space. </br>
# *   Display the reconstructed image along with the original image.
# *   Report the reconstruction mean squared error. 

# In[21]:


def reconstruct(randnum,pca,X_pca,h,w):
  randimg = X[randnum]
  eigen_weights = X_pca.transform(randimg.reshape(1,-1))
  recon_pixels= pca.inverse_transform(eigen_weights)
  recon_image = reshape(recon_pixels,(h,w))
  mean_squared_error = ((randimg.reshape(h,w)-recon_image)**2).mean()
  print("mean squared error:",mean_squared_error)
  figure()
  imshow(recon_image,cmap =cm.bone,interpolation = 'nearest')
  title("Reconstructed Image")
  figure()
  imshow(randimg.reshape(h,w),cmap=cm.bone,interpolation='nearest')
  title("Original Image")
  show()

reconstruct(170,pca,X_pca,h,w)


# Now that you have a reduced-dimensionality vector, train a single hidden layer neural network classifier with the person names as outputs and the reduced image vectors as input. You can use scikit-learn's MLPClassifier, with the number of neurons in the hidden layer set to 1024. 

# In[22]:


def mlpclassifier(X_train_pca,y_train,X_test_pca):
  NN=MLPClassifier(hidden_layer_sizes=(1024,))
  model=NN.fit(X_train_pca,y_train)
  y_pred = model.predict(X_test_pca)
  return model,y_pred 

model,y_pred=mlpclassifier(X_train_pca,y_train,X_test_pca)


# Now test your network's predictions on the test set and print out the precision, recall and support values for each class. Also display the images (say, 10 test images) along with the true and the  predicted label.

# In[23]:


def head(y_pred,y_test,target_names,i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def plot_compare(X_test,y_test,y_pred):
  print(classification_report(y_test,y_pred,target_names=target_names))
  prediction_titles = [head(y_pred,y_test,target_names,i) for i in range(y_pred.shape[0])]
  plot_gallery(X_test, prediction_titles, h, w)

plot_compare(X_test,y_test,y_pred)


# Repeat all the above steps for different values of number of principal components or eigen faces - 100, 150 and 200. Explain the change in the proportion of the total variance explained by the eigenfaces and reconstruction mean squared error with increase in the number of eigenfaces.

# In[27]:


def face_recognition(n_comp,X_train,y_train,X_test,y_test,h,w):
  pca,X_pca,eigenfaces,X_train_pca,X_test_pca=extract_pc(n_comp,X_train,X_test,h,w)
  #plot_eigenfaces(n_comp,eigenfaces,h,w)                         # uncomment for eigenfaces
  reconstruct(170,pca,X_pca,h,w)
  model,y_pred=mlpclassifier(X_train_pca,y_train,X_test_pca)
  #plot_compare(X_test,y_test,y_pred)                             # uncomment for true and predicted classes

face_recognition(100,X_train,y_train,X_test,y_test,h,w)
face_recognition(150,X_train,y_train,X_test,y_test,h,w)
face_recognition(200,X_train,y_train,X_test,y_test,h,w)


# Variance is almost same for three cases that is in 1:1:1 proportion and Mean squared error decreases as the number of principal components increases

# Bonus : Try using eigenfaces to recognize images of animal faces.
# 
