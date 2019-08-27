from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmod_activation(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict(X,W):
    preds = sigmod_activation(X.dot(W))

    # apply a step function to threshold the outputs to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds

def next_batch(X, y, batchSize):
    for i in np.arange(0, X.shape[0], batchSize):
        yield(X[i:i+batchSize], y[i:i+batchSize])

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.001, help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of SGD mini-batches")

args = vars(ap.parse_args())

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

X = np.c_[X, np.ones(X.shape[0])]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO] training ...")
W = np.random.rand(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    # initialize the total loss for the epoch
    epochLoss = []

    #loop over our data in batches
    for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
        preds = sigmod_activation(batchX.dot(W))
        error = preds - batchY 
        eloss = np.sum(error ** 2)
        epochLoss.append(eloss)

        gradient = batchX.T.dot(error)

        W += -args["alpha"] * gradient

    loss = np.average(epochLoss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch+1), loss))

# evaluate our model
print("[INFO] evaluating ...")
preds = predict(testX, W)
print(classification_report(testY, preds))

## plot the (testing) classification data
#plt.style.use("ggplot")
#plt.figure()
#plt.title("Data")
#plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)
#
## construct a figure that plots the loss over time
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0, args["epochs"]), losses)
#plt.title("Training Loss")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss")
#plt.show()
