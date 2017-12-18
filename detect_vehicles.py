'''
Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
Estimate a bounding box for vehicles detected.
'''

import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from glob import glob
from moviepy.editor import VideoFileClip

from features import *


heat_threshold = 0
heatmaps = []


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range):
    #draw_img = np.copy(img)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, cspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    channels = []
    if hog_channel == 'ALL':
        for channel in range(ctrans_tosearch.shape[2]):
            channels.append(ctrans_tosearch[:, :, channel])
    else:
        channels.append(ctrans_tosearch[:, :, hog_channel])

    # Define blocks and steps as above
    nxblocks = (channels[0].shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (channels[0].shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hogs = []
    for channel in channels:
        hogs.append(get_hog_features(channel, orient, pix_per_cell, cell_per_block, feature_vec=False))

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feats = []
            for hog in hogs:
                hog_feats.append(hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel())

            hog_features = np.hstack(hog_feats)

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins, bins_range=hist_range)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))

            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                #cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                # Generate a heatmap of detections for this image
                box = [[xbox_left, ytop_draw + ystart],[xbox_left + win_draw, ytop_draw + win_draw + ystart]]
                heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heat

# Returnd a copy of image with bounding boxes labeled
def draw_labeled_bboxes(image, labels):
    img = np.copy(image)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def process_image(img):
    # Get a heatmap of detected cars
    heatmaps.append(find_cars(img, ystart, ystop, scale, svc, X_scaler, colorspace, orient, pix_per_cell, cell_per_block,
                     hog_channel, spatial_size, hist_bins, hist_range))

    if len(heatmaps) > heat_smoothing:
        heatmaps.pop(0)

    heat = np.sum(heatmaps, axis=0)

    # Apply a threshold to the detections heatmap and zero out pixels below the threshold
    heat[heat <= heat_threshold] = 0

    # Find final boxes from heatmap using label function
    labels = label(heat)
    return draw_labeled_bboxes(img, labels)


def test_images():
    global heatmaps, heat_threshold
    heat_threshold = 2

    # Get image filenames
    img_filenames = glob("./test_images/test*.jpg")

    for img_filename in img_filenames:
        # Load test image
        img = mpimg.imread(img_filename)
        # Clear heatmap
        heatmaps = []
        # Process image and write result
        mpimg.imsave("./output_images/" + img_filename.split("/")[-1], process_image(img))


def test_video():
    global heatmaps, heat_threshold
    heat_threshold = 50
    # Clear heatmap
    heatmaps = []
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip = VideoFileClip("./project_video.mp4").subclip(39, 42)
    clip = VideoFileClip("./test_video.mp4")
    #clip = VideoFileClip("./project_video.mp4")

    # Process the video
    result = clip.fl_image(process_image)  # NOTE: this function expects color images!!

    # Save the processed video
    result.write_videofile("./output_images/output_video.mp4", audio=False)





# Load trained model and feature parameters from pickle
dist_pickle = joblib.load('svc.p')

svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
colorspace = dist_pickle["colorspace"]

orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
hog_channel = dist_pickle["hog_channel"]

spatial_size = dist_pickle["spatial_size"]

hist_bins = dist_pickle["hist_bins"]
hist_range = dist_pickle["hist_range"]

# Set search parameters
ystart = 400
ystop = 656
scale = 1.5

heat_smoothing = 15

# Load test image
# img = mpimg.imread('./test_images/test1.jpg')

test_images()
#test_video()

# plt.imshow(draw_img)
# plt.show()
#
# # Visualize the heatmap when displaying
# heatmap = np.clip(heat, 0, 255)
#
# plt.imshow(heatmap, cmap='hot')
# plt.show()

