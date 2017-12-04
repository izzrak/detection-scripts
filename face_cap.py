import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './checkpoints/mobimod.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './datasets/wider_label_map.pbtxt'

NUM_CLASSES = 1
category_index = {1: {'id': 1, 'name': 'face'}}
class_name ='face'
red = (0,0,255,255)
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(2, 3) ]
print(TEST_IMAGE_PATHS)

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
			(im_height, im_width, 3)).astype(np.uint8)

'''
def draw_bounding_box_on_image_array(image,
									ymin,
									xmin,
									ymax,
									xmax,
									color='red',
									thickness=4,
									display_str_list=(),
									use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box in normalized coordinates (same below).
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
	image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
	draw = ImageDraw.Draw(image)
	im_width, im_height = image.size
	if use_normalized_coordinates:
		(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
	else:
		(left, right, top, bottom) = (xmin, xmax, ymin, ymax)
 	draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
	try:
		font = ImageFont.truetype('arial.ttf', 24)
	except IOError:
		font = ImageFont.load_default()

		text_bottom = top
	# Reverse list and print from bottom to top.
	for display_str in display_str_list[::-1]:
 		text_width, text_height = font.getsize(display_str)
		margin = np.ceil(0.05 * text_height)
		draw.rectangle(
				[(left, text_bottom - text_height - 2 * margin), 
				(left + text_width, text_bottom)],fill=color)
		draw.text(
				(left + margin, text_bottom - text_height - margin),
				display_str, fill='black', font=font)
		text_bottom -= text_height - 2 * margin
	np.copyto(image, np.array(image_pil))

'''
def visualize_prediction(image, boxes, classes, scores):
	print(boxes.shape, scores.shape, classes.shape, num)
	box_num = boxes.shape[0]
	for i in range(box_num):
		if scores[i] < 0.5:
			break
		else:
			draw_bounding_box_on_image(image, boxes[i], scores[i])
	return image
def draw_bounding_box_on_image(image, box, score):
	'''
	# draw bounding box
	ymin, xmin, ymax, xmax = box
	height, width, depth = image.shape
	(left, right, top, bottom) = (int(xmin * width), int(xmax * width),
									int(ymin * height), int(ymax * height))
	points = np.array([(left, top), (left, bottom), (right, bottom), (right, top)])
	cv2.polylines(image, np.int32([points]), isClosed=True, color=red, thickness=4)
	'''
	# draw label and score
	display_str = '{}: {}%'.format(class_name, int(100*score))
	cv2.rectangle(image, (left, top), (right, bottom), color=red, thickness=cv2.CV_FILLED)
	
	print(display_str)

	return

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(graph=detection_graph, config=config) as sess:
		# Definite input and output Tensors for detection_graph
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		for image_path in TEST_IMAGE_PATHS:
			image = cv2.imread(image_path)
			print(image.shape)
			#image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			#image_np = load_image_into_numpy_array(image)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image, axis=0)
			# Actual detection.
			(boxes, scores, classes, num) = sess.run(
					[detection_boxes, detection_scores, detection_classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})
			print(boxes.shape, scores.shape, classes.shape, num)
			image_pred = visualize_prediction(image, np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32), np.squeeze(scores))
			
			#cv2.imshow('img', image_pred)
			input()