# This is a simmple pedestrian detection prgogram
# using built-in openCV object detector
# and on top an SVM to classify pedestrian 

# adapted from http://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

import os
import numpy as np
import cv2
# A series of openCV function in a python API
import imutils
from imutils import paths
from imutils.object_detection import non_max_suppression

from parse import *

# initialize the object detector
hog = cv2.HOGDescriptor()
# plug on top an SVM to only get people
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Quantitative test are going to be done on the Penn Fudan Ped dataset
# containing around 100 images

# On the PennFudanPed dataset we have access to the bounding boxes of the 
# pedestrian

# We retrieve for each image the bounding boxes
# we store them in a dictionary
BoundingBoxes = {}

path_annotations = []
for dirname, dirnames, filenames in os.walk('./Annotation'):
    # print path to all filenames.
    for filename in filenames:
    	#path_annotations.append(os.path.join(dirname, filename)) 
    	with open(os.path.join(dirname, filename)) as annotation:
    		#BoundingBoxes[filename] = []
    		coordinates = []
    		for line in annotation:
				if(line[0:20] == "Bounding box for obj"):
					print line
					s = search("({:d}, {:d}) - ({:d}, {:d})",line)
					coordinates.append([s[0],s[1],s[2],s[3]])
		# append the coordinates of bounding box to the dictionnary
		BoundingBoxes[filename] = coordinates


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max((xB - xA + 1) * (yB - yA + 1),0)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	if(float(boxAArea + boxBArea - interArea) > 0):
		iou = interArea / float(boxAArea + boxBArea - interArea)
	else:
		return 0

 
	# return the intersection over union value
	return iou

def iou_several_boxes(boxesA, boxesB):
	cum_iou = 0  # we will add the cumulative intersection ever union here
	nA = len(boxesA)
	nB = len(boxesB)
	for boxA in boxesA:
		cur_iou = 0
		for boxB in boxesB:
			iou =  bb_intersection_over_union(boxA, boxB)
			if(iou > cur_iou):
				cur_iou = iou
		# cur_iou is now to the max of the iou possible related to ground truth values
		cum_iou += cur_iou

	if(nA > 0):
		return cum_iou/nA
	else:
		return 0


iou_dict = {}

for key, value in BoundingBoxes.iteritems():
	# we load the image
	image = cv2.imread("./PNGImages/" + key[:-4] + ".png")
	# resize it
	# image = imutils.resize(image, width=min(400, image.shape[1]))
	original_image = image.copy()

	# launch the detection
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
	print(rects)
	print(weights)

	# # draw the bounding boxes on the original image
	# for (x, y, w, h) in rects:
	# 	cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# qualitative results
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(original_image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	for (x1, y1, x2, y2) in value:
		cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)		
	cv2.imshow("Before NMS", original_image)

	# quantitative results
	iou_dict[key] = iou_several_boxes(pick, value)

	# wait for user input to process next image
	if((iou_several_boxes(pick, value) > 0.4) append (iou_several_boxes(pick, value) < 0.45)):
	 	cv2.waitKey(0)

print(iou_dict)
mean_accuracy = 0
for k,v in iou_dict.iteritems():
	mean_accuracy += v
print(mean_accuracy/len(iou_dict))
