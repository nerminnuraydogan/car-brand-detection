import cv2
import numpy as np

import time
import sys
import os

# adapted from https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

# Minimum probability to filter weak detections.
# a default value of 50% (0.5 ),
# but you should feel free to experiment with this value.
CONFIDENCE = 0.5

# non-maxima suppression threshold with a default value of 0.5 .
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# the neural network configuration
config_path = "cfg/yolo-obj.cfg"

# the YOLO net weights file
weights_path = "weights/yolo-obj_final.weights"

# loading all the class labels (objects)
LABELS = open("data/obj.names").read().strip().split("\n")

# generating colors for each object for later plotting
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


# load the YOLO object detector trained on the custom dataset (10 classes)
# and determine only the *output* layer names that is needed from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# the default saving video file extension for this code is avi (for Windows)
# this code is tested with mp4 extension videos

# change path name to experiment with videos
path_name = "data/mercedes.mp4"
output_path = "output_video/mercedes.avi"


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(path_name)
writer = None
(W, H) = (None, None)


# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, in return will be giving the bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize the lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,
		SCORE_THRESHOLD)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output_path, fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
