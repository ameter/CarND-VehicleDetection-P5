'''
Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
Estimate a bounding box for vehicles detected.
'''

import numpy as np
import cv2

import matplotlib.image as mpimg
from sklearn.externals import joblib
from scipy.ndimage.measurements import label, find_objects
from glob import glob
from moviepy.editor import VideoFileClip
import os

from features import *

DEBUG = True

vehicles = []
smoothing_factor = 1
frame = 0


# Define a class to receive the characteristics of each vehicle detection
class Vehicle:
    def __init__(self, position):

        # Store vehicle positions
        self.positions = [position]

        # Store frames since seen
        self.frames_since_seen = 0

    # Compute mean position
    def mean_position(self):
        sum_x_start = 0
        sum_x_stop = 0
        sum_y_start = 0
        sum_y_stop = 0

        for pos in self.positions:
            sum_x_start += pos["x"].start
            sum_x_stop += pos["x"].stop
            sum_y_start += pos["y"].start
            sum_y_stop += pos["y"].stop

        mean_pos = {
            "x": {
                "start": sum_x_start / len(self.positions),
                "stop": sum_x_stop / len(self.positions)
            },
            "y": {
                "start": sum_y_start / len(self.positions),
                "stop": sum_y_stop / len(self.positions)
            },
        }
        return mean_pos

    # Compute mean size
    def mean_size(self):
        pos = self.mean_position()
        size = (pos["x"]["stop"] - pos["x"]["start"]) * (pos["y"]["stop"] - pos["y"]["start"])
        return size



# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range):
    # Set search parameters
    ystart = 350
    ystop = 656
    scale = 1.5

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


def update_vehicle_positions(heatmap):
    global vehicles

    #output_heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    #output_heatmap[detection] = heat

    # Increment frames since seen for all vehicles
    for vehicle in vehicles:
        vehicle.frames_since_seen += 1

    # Drop vehicle if not seen in past n frames, where n = smoothing factor
    vehicles[:] = [vehicle for vehicle in vehicles if not vehicle.frames_since_seen >= smoothing_factor]


    # Apply a first-level threshold to the detections heatmap and zero out pixels below the threshold
    #heatmap[heatmap <= 1] = 0
    heatmap_top = heatmap[:480, :]
    heatmap_bottom = heatmap[480:, :]
    heatmap_top[heatmap_top <= 1] = 0
    heatmap_bottom[heatmap_bottom <= 6] = 0
    heatmap = np.append(heatmap_top, heatmap_bottom, axis=0)

    # Find boxes from heatmap using label function
    labels = label(heatmap)
    detections = find_objects(labels[0])

    # Filer heatmap based on heat factor of each box.
    for detection in detections:
        x = detection[1]
        y = detection[0]
        size = (x.stop - x.start) * (y.stop - y.start)
        heat = np.sum(heatmap[detection])

        #heat_factor = (((heat * 1.0) + (size * 10.0)) - (y.stop * 5000.0))
        heat_factor = ((heat * 30.0) - (y.stop * 1000.0)) + 150000.0

        # See if we are already tracking a vehicle associated with this detection
        matched_vehicles = []
        for vehicle in vehicles:
            vehicle_pos = vehicle.mean_position()
            if x.start < vehicle_pos["x"]["stop"] and x.stop > vehicle_pos["x"]["start"] and y.start < vehicle_pos["y"]["stop"] and y.stop > vehicle_pos["y"]["start"]:
                # detection overlaps with the vehicle
                vehicle_size = vehicle.mean_size()
                # Get percentage change in size of detection from size of vehicle
                size_change = abs(size - vehicle_size) / vehicle_size
                print("size change", size_change)
                # if size_change > .20:
                matched_vehicles.append(vehicle)

        if len(matched_vehicles) == 1:
            vehicle = matched_vehicles[0]
            # Detection matched exactly one vehicle, treat as an update
            # Apply lessor threshold filtering for heat
            if heat_factor > 0:
                if DEBUG: print("\nframe:", frame, "old heat:", heat_factor, "kept")
                # Add current detection to vehicle's position list
                vehicle.positions.append({"x": x, "y": y})
                if len(vehicle.positions) > smoothing_factor:
                    vehicle.positions.pop(0)
                vehicle.frames_since_seen = 0
            else:
                if DEBUG: print("\nframe:", frame, "old heat:", heat_factor, "dropped")
        elif len(matched_vehicles) == 0:
            # Detection did not match a vehicle, treat as new
            # Note, we are dropping detections that match multiple vehicles
            # Apply stricter threshold filtering for heat
            if heat_factor > 0:
                if DEBUG: print("\nframe:", frame, "new heat:", heat_factor, "kept")
                # Add current detection to vehicle's position list
                vehicles.append(Vehicle({"x": x, "y": y}))
            else:
                if DEBUG: print("\nframe:", frame, "new heat:", heat_factor, "dropped")


    # # Store filtered heatmap
    # heatmaps.append(output_heatmap)
    # if len(heatmaps) > heat_smoothing:
    #     heatmaps.pop(0)
    #
    # # Get heatmap that's the mean of stored heatmaps
    # heatmap = np.mean(heatmaps, axis=0)
    #
    # # Third-level filter
    # # Apply a threshold to the detections heatmap and zero out pixels below the threshold
    # heatmap[heatmap <= 1] = 0
    #
    # labels = label(heatmap)
    # detections = find_objects(labels[0])
    # for detection in detections:
    #     heat = heatmap[detection]
    #
    #     x = detection[1]
    #     y = detection[0]
    #
    #     #heat_factor = (np.sum(heat) - (y.stop * 1000)) + 490000
    #     #heat_factor = (np.sum(heat) - (y.stop * 1000)) + 500000
    #     heat_factor = np.sum(heat)
    #
    #     if DEBUG:
    #         print("\nframe:", frame, "heat2:", heat_factor)




# Returnd a copy of image with bounding boxes labeled
def draw_bounding_boxes(image):
    img = np.copy(image)

    for vehicle in vehicles:
        # If we're smoothing, ensure we've had more than one detection for the vehicle
        if smoothing_factor > 1:
            if len(vehicle.positions) < 5:
                continue

        # Get the mean position for the vehicle
        position = vehicle.mean_position()

        # Define a bounding box based on min/max x and y and draw the box on the image
        cv2.rectangle(img, (int(position["x"]["start"]), int(position["y"]["start"])), (int(position["x"]["stop"]), int(position["y"]["stop"])), (0, 0, 255), 6)

    if DEBUG: mpimg.imsave("./output_images/frame" + str(frame) + ".jpg", img)

    # Return the image
    return img



def process_image(img):
    global frame
    frame += 1

    # Get a heatmap of detected vehicles
    heatmap = find_cars(img, svc, X_scaler, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range)

    # Update vehicle positions
    update_vehicle_positions(heatmap)

    # Draw vechicle boxes
    return draw_bounding_boxes(img)


def test_images():
    global vehicles, smoothing_factor
    smoothing_factor = 1

    # Get image filenames
    img_filenames = glob("./test_images/test*.jpg")

    for img_filename in img_filenames:
        # Load test image
        img = mpimg.imread(img_filename)

        # Clear vehicles
        vehicles = []

        # Process image and write result
        mpimg.imsave("./output_images/" + img_filename.split("/")[-1], process_image(img))


def test_video():
    global vehicles, smoothing_factor
    smoothing_factor = 10

    # Clear vehicles
    vehicles = []

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds

    #clip = VideoFileClip("./project_video.mp4").subclip(25, 28)
    #clip = VideoFileClip("./project_video.mp4").subclip(45, 50)
    clip = VideoFileClip("./project_video.mp4").subclip(5, 12)

    #clip = VideoFileClip("./test_video.mp4")
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



test_images()
frame = 0
test_video()


# Play a sound when done (Mac OS specific file location)
os.system("open /System/Library/Sounds/Glass.aiff")


