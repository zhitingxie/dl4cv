# import the neccessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# contruct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--datasets", required=True, help="path to input datasets")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# initialize the class labels
classLabels = ["cat", "dog", "panda"]

# grab the list of images in the dataset then randomly sample indexes into the image paths list
print("[INFO] sampling images ...")
imagePaths = np.array(list(paths.list_images(args["datasets"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# initialize the image preprocessor
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors = [sp, iap])
(data, labels) = sdl.load(imagePaths, verbose = 500)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] loading pre-trained network ...")
model = load_model(args["model"])

# evaluate the network
print("[INFO] evaluating network ...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the predictions, and display it to our screen
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

## plot the training loss and accuracy
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
#plt.title("Training Loss and Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend()
#plt.show()

