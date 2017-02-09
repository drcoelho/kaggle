import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes, neural_network, svm
from random import randint
import warnings

# Use it in notebook
# %matplotlib inline
warnings.filterwarnings("ignore", category=DeprecationWarning)

labeled_images = pd.read_csv('./data/train.csv')

images = labeled_images.iloc[0:100,1:]
labels = labeled_images.iloc[0:100,:1]

img = images.iloc[0].as_matrix()
print img.shape
img = img.reshape((28, 28))
print img.shape


n = 1
sample = images.sample(n=n)
print sample.head()
for index in range(n):
    img = sample.iloc[index].as_matrix()
    print len(img), img
    img = img.reshape((28, 28))
    print len(img), img

    plt.imshow(img, cmap='gray')
    plt.show()


print len(images), len(labels)

# Split for cross validation
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8)
# test_images[test_images>0]=1
# train_images[train_images>0]=1


def show_img(dataset, index):
    # Show an image example
    img = dataset.iloc[index].as_matrix()
    img = img.reshape((28,28))
    plt.imshow(img, cmap='gray')
    plt.title(train_labels.iloc[index, 0])
    plt.show()

# Training SVM model
# clf = naive_bayes.GaussianNB()
# clf.fit(train_images, train_labels.values.ravel())
# print clf.score(test_images,test_labels)


# #
# clf = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(train_images, train_labels.values.ravel())
# print clf.score(test_images,test_labels)


#
clf = svm.SVC(kernel='linear')
#clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print clf.score(test_images,test_labels)

validation = pd.read_csv('./data/test.csv')
exit(0)
for index in range(1, 10):
    print clf.predict(validation.iloc[index])
    #print clf.predict(test_images.iloc[index]), test_labels.iloc[index][0]
    show_img(validation, index)


