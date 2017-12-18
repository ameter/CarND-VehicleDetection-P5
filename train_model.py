'''
Reads in a labeled training set of images and trains a linear SVM classifier.
Extracts binned color, histogram of color, and histogram of oriented gradients (HOG) features.
Appends and normalizes features and randomizes a selection for training and testing.
Saves trained model to a pickle file for later use.
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.externals import joblib

from features import *

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, cspace)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Append the new feature vector to the features list
        features.append(np.concatenate((hog_features, spatial_features, hist_features)))

    # Return list of feature vectors
    return features




# Divide up into cars and notcars
cars = glob.glob("/Users/chris/Developer/CarND-VehicleDetection-P5/data/vehicles/*/*.png")
notcars = glob.glob("/Users/chris/Developer/CarND-VehicleDetection-P5/data/non-vehicles/*/*.png")

# Set tunable parameters
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# HOG Parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
# Spatial Binning of Color Parameters
spatial_size = (32, 32)
# Color Histogram Parameters
hist_bins = 32
hist_range = (0, 256)

# Extract all features
print('\nExtracting features with the following parameters...\n\tColorspace:\n\t\t', colorspace, '\n\tHOG:\n\t\t', orient, 'orientations\n\t\t', pix_per_cell, 'pixels per cell\n\t\t', cell_per_block, 'cells per block\n\t\t', hog_channel, 'channel\n\tColor Histogram:\n\t\t', hist_bins, 'bins\n\t\t', hist_range, 'range\n\tColor Spatial Binning:\n\t\t', spatial_size, 'size')

t=time.time()
car_features = extract_features(cars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)

t2 = time.time()
print(round(t2-t, 2), 'seconds to extract features.')

# Normalize features
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('\nTraining linear SVM classifier...')
print('\nFeature vector length:', len(X_train[0]))

# Train the model
# Use a linear SVC
#parameters = {'C':[1, 10]}
#svr = LinearSVC()
#svc = GridSearchCV(svr, parameters)
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'seconds to train SVC.')

# Check the score of the SVC
print('\nTest Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('\nSVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'seconds to predict', n_predict, 'labels with SVC')

# Save the trained model and all feature parameters
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler

dist_pickle["colorspace"] = colorspace

dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["hog_channel"] = hog_channel

dist_pickle["spatial_size"] = spatial_size

dist_pickle["hist_bins"] = hist_bins
dist_pickle["hist_range"] = hist_range

# Save using scikit's replacement of pickle, which is more efficient on objects that carry large numpy arrays internally
joblib.dump(dist_pickle, 'svc.p')
print('\nModel saved.')


