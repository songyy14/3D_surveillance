import os
import sys
import copy
import random
sys.path.append('trinet')

import cv2
import numpy as np
import matplotlib.pyplot as plt

from trinet.inference import euclidean_dist, net_input_width, net_input_height

# IOU threshold used to determine whether to initialize new tracker
# at the particular position.
IOU_th = 0.1 / 1.9

# Spatial distance threshold. If distance between two persons viewd
# in different cameras is above the threshold, they are very likely to
# be different persons.
spatial_dist_th = 5


# Spatial constraints.

def read_camera_matrix(path):
	camera_matrix = np.zeros((3, 4))
	with open(path, 'r') as file:
		for i in range(3):
			line = file.readline()
			splits = line.split()
			for j in range(4):
				camera_matrix[i, j] = float(splits[j])

	return camera_matrix


def intersect_plane(camera_matrix, u, v, plane):
	'''
	Back-project one point in the image onto a plane in the world coordinate
		system.

	Args:
		camera_matrix: Camera matrix transforming world coordinate to pixel
			coordinate. Numpy array of shape (3, 4).
		u, v: Pixel coordinate of the point.
		plane: A vector representing the equation of the plane. Numpy array of
			shape (4,).

	Returns:
		coord_w: World coordinate of the point, where the ray originating
			at camera center and traveling through pixel (u, v) intersects
			the plane. Numpy array of shape (3,).
	'''
	# Preparation.
	A = np.zeros((4, 4))
	A[0:3, 0:3] = camera_matrix[0:3, 0:3]
	A[3, 0:3] = plane[:3]
	A[:, 3] = [-u, -v, -1, 0]

	b = np.zeros((4, 1))
	b[0:3, 0] = - camera_matrix[0:3, 3]
	b[3, 0] = - plane[3]

	# Get result.
	solution = np.linalg.inv(A) @ b
	coord_w = np.squeeze(solution[0:3, 0])

	return coord_w


def get_blacklist(u, v, this_cam, trackers, camera_matrices):
	'''
	Find those persons tracked by other cameras who are far from
	the interested person on the ground plane. And they are likely
	to be persons different from the interested one.

	Args:
		u, v: Pixel coordinate of the interested person.
		this_cam: Name of the camera capturing the person.
		trackers: Trackers of all cameras.
		camera_matrices: Camera matrices of all cameras.
	'''
	ground_plane = np.array([0, 0, 1, 0])
	coord0 = intersect_plane(camera_matrices[this_cam], u, v, ground_plane)
	coord0 = np.expand_dims(coord0, axis=0)

	pid_list = []
	coord_list = []
	for cam in trackers:
		if cam == this_cam:
			continue
		for tracker in trackers[cam]:
			x, y, w, h = tracker.get_roi()
			coord_list.append(intersect_plane(camera_matrices[cam], x + w/2, y + h, ground_plane))
			pid_list.append(tracker.pid)

	if len(coord_list) == 0:
		return []

	coords = np.stack(coord_list, axis=0)
	dist_matrix = euclidean_dist(coord0, coords)
	blacklist = []
	for j in range(dist_matrix.shape[1]):
		if dist_matrix[0, j] > spatial_dist_th:
			blacklist.append(pid_list[j])

	return blacklist


# Combine detection and tracking result.

def IOU(rect1, rect2):
	'''
	Calculate intersect-of-union of two rectangles.

	Args:
		rect1, rect2: Numpy array of shape (4,), representing a rectangle
			in the form of (x, y, w, h).
	'''
	xmin = max(rect1[0], rect2[0])
	ymin = max(rect1[1], rect2[1])
	xmax = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
	ymax = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

	intersect = max(xmax - xmin, 0) * max(ymax - ymin, 0)
	iou = intersect / (rect1[2] * rect1[3] + rect2[2] * rect2[3] - intersect)
	return iou


def detection_query(detect_pos, track_pos):
	'''
	Query the person-detection result of one frame, and compare it with
		current tracking result, to determine where new trackers should
		be added.
	
	Args:
		detect_pos, track_pos: Numpy array of shape (n, 4), each row of which 
			represents (x, y, w, h) of the bounding box.

	Returns:
		indices: A list of Indices of those detection that will be used to
			initialize new trackers.
	'''
	indices = []
	for i in range(detect_pos.shape[0]):
		to_add = True
		for j in range(track_pos.shape[0]):
			if IOU(detect_pos[i, :], track_pos[j, :]) > IOU_th:
				to_add = False
				break

		if to_add:
			indices.append(i)

	return indices


# Observe distribution of distance between images.

def get_pid(image_name):
	return int(image_name.split('_')[1])


def group_images(image_names):
	'''
	Put images of the same person together.
	
	Args:
		path_images: A list of image names.

	Returns:
		pid2names: A dict mapping person-id to a list of names of all images
			that belong to this person.
	'''
	pid2names = dict()
	for name in image_names:
		pid = get_pid(name)
		if pid in pid2names:
			pid2names[pid].append(name)
		else:
			pid2names[pid] = [name]

	return pid2names


def plot_hist(samples, bins):
	'''
	Plot histogram of samples.

	Args:
		samples: A list of samples.
	'''
	hist, bin_edges = np.histogram(samples, bins=bins, density=True)
	n = hist.shape[0]
	bin_mids = np.zeros((n,))
	for i in range(n):
		bin_mids[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

	plt.plot(bin_mids, hist)
	# plt.show()


def distance_hist(features1, images1, features2, images2, intra_camera):
	'''
	Plot histogram of intra-person distance as well as extra-person distance.
	The image pair are chosen from the same camera if intra_camera is true, 
	otherwise from different cameras.

	features1, features2: Numpy array of shape (n, 128), each row of which
		corresponds to one image.
	images1, images2: List of image names (xxx.jpg\n).
	'''
	idistance = []
	edistance = []
	# First, calculate distance of each image pair.
	# Then traverse the distance matrix, determine whether the pair belongs to
	# the same person, and append the distance to one of the lists accordingly.
	if intra_camera:
		features_op = [features1, features2]
		images_op = [images1, images2]
		for c in [0, 1]:
			features = features_op[c]
			images = images_op[c]
			dist_matrix = euclidean_dist(features, features)
			for i in range(len(images)):
				for j in range(len(images)):
					if i <= j:
						continue
					pid_i = images[i].split('_')[1]
					pid_j = images[j].split('_')[1]
					if pid_i == pid_j:
						idistance.append(dist_matrix[i, j])
					else:
						edistance.append(dist_matrix[i, j])

	else:
		dist_matrix = euclidean_dist(features1, features2)
		for i in range(len(images1)):
			for j in range(len(images2)):
				pid_i = images1[i].split('_')[1]
				pid_j = images2[j].split('_')[1]
				if pid_i == pid_j:
					idistance.append(dist_matrix[i, j])
				else:
					edistance.append(dist_matrix[i, j])

	print('intra-person: %d' % len(idistance))
	print('extra-person: %d' % len(edistance))

	# bins = 10
	# plot_hist(idistance, bins)
	# plot_hist(edistance, bins)
	# plt.show()

	X = np.linspace(min(idistance + edistance), max(idistance + edistance), 1000)
	sigma = 1
	Yi = parzen_window(idistance, sigma, X)
	Ye = parzen_window(edistance, sigma, X)
	plt.plot(X, Yi)
	plt.plot(X, Ye)
	plt.show()


def gaussian_func(X, mu, sigma):
	Y = np.exp(- (X - mu)*(X - mu) / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma)
	return Y


def parzen_window(samples, sigma, X):
	'''
	Estimate probability density using limited number of samples.

	Args:
		samples: A list of samples.
		sigma: Parameter of normal distribution.
		X: Numpy 1D-array, specifying where to evaluate the density function.
	'''
	array_sum = np.zeros(X.shape)
	for i in range(len(samples)):
		array_sum = array_sum + gaussian_func(X, samples[i], sigma)

	return array_sum / len(samples)


def min_distance_hist(features1, images1, features2, images2, intra_camera):
	'''
	A modified version of `distance_hist`.
	Plot histogram of the minimum distance between each probe image and gallery images.
	The probe is one image at a time. Gallery images contain random number of images
	of unique persons, with the probe itself excluded. Repeated experiments are performed.

	Returns:
		avg_min_distance: Average of all observed minimum distance, regardless of intra-person
			or extra-person.
	'''
	repeat = 100

	idistance = []
	edistance = []
	# person-id are shared for the two image sets.
	pids = [pid for pid in group_images(images1)]
	
	features_op = [features1, features2]
	images_op = [images1, images2]
	
	for c in [0, 1]:
		features_p = features_op[c]
		images_p = images_op[c]
		if intra_camera:
			features_g = features_op[c]
			images_g = images_op[c]
		else:
			features_g = features_op[1 - c]
			images_g = images_op[1 - c]
		
		pid2names = group_images(images_g)

		# Calculate and sort distance matrix in advance.
		dist_matrix = euclidean_dist(features_p, features_g)
		indices = np.argsort(dist_matrix, axis=1)

		# Traverse `images` to get one probe.
		for probe in images_p:
			probe_pid = get_pid(probe)
			probe_index = images_p.index(probe)
			# Repeat randomly sampling gallery images.
			for i in range(repeat):
				gallery_indices = []
				gallery_pids = random.sample(pids, random.randint(1, len(pids)))
				for gpid in gallery_pids:
					tmp = copy.deepcopy(pid2names[gpid])
					if (probe in tmp) and intra_camera:
						tmp.remove(probe)
					gallery = random.choice(tmp)
					gallery_indices.append(images_g.index(gallery))

				# Find the minimum distance.
				for j in range(indices.shape[1]):
					if indices[probe_index, j] in gallery_indices:
						break
				index_min = indices[probe_index, j]
				pid_min = get_pid(images_g[index_min])
				dist_min = dist_matrix[probe_index, index_min]
				if pid_min == probe_pid:
					idistance.append(dist_min)
				else:
					edistance.append(dist_min)

	print('intra-person: %d' % len(idistance))
	print('extra-person: %d' % len(edistance))

	# bins = 10
	# plot_hist(idistance, bins)
	# plot_hist(edistance, bins)

	X = np.linspace(0, 30, 1000)
	sigma = 1
	Yi = parzen_window(idistance, sigma, X)
	Ye = parzen_window(edistance, sigma, X)
	plt.plot(X, Yi)
	plt.plot(X, Ye)
	plt.show()

	return np.mean(np.array(idistance + edistance))


def spatial_dist_hist(path_coords1, path_coords2):
	'''
	Plot histogram of spatial distance of each two persons captured in the same frame
	of different cameras. The spatial coordinates are estimated by casting a ray from
	camera center and finding its intersection with ground plane.
	'''
	idistance = []
	edistance = []

	# Travel through all frames and collect data.
	with open(path_coords1, 'r') as file1, open(path_coords2, 'r') as file2:
		# Jump over the first header.
		file1.readline()
		file2.readline()
		to_terminate = False
		while not to_terminate:
			# Read files.
			file_op = [file1, file2]
			coordinate_list = [[], []]
			pids = [[], []]
			for i in range(2):
				while True:
					line = file_op[i].readline()
					if line == '':
						to_terminate = True
						break
					elif line[0] == '#':
						break
					splits = line.split()
					coordinate_list[i].append([float(splits[0]), float(splits[1])])
					pids[i].append(int(splits[2]))

			coordinates1 = np.array(coordinate_list[0])
			coordinates2 = np.array(coordinate_list[1])
			pids1 = pids[0]
			pids2 = pids[1]

			# Calculate distance of two persons each from one camera.
			dist_matrix = euclidean_dist(coordinates1, coordinates2)
			for i in range(len(pids1)):
				for j in range(len(pids2)):
					if pids1[i] == pids2[j]:
						idistance.append(dist_matrix[i, j])
					else:
						edistance.append(dist_matrix[i, j])

	print('intra-person: %d' % len(idistance))
	print('extra-person: %d' % len(edistance))

	# Plot histogram.
	# plot_hist(idistance, 10)
	# plot_hist(edistance, 10)
	# plt.xlim(xmin = 0)
	# plt.ylim(ymin = 0)
	# plt.show()


def foo(path_coords1, path_coords2):
	'''
	Plot histogram of spatial distance of each two persons captured in the same frame
	of different cameras. The spatial coordinates are estimated by casting a ray from
	camera center and finding its intersection with ground plane.
	'''
	coords = []

	# Travel through all frames and collect data.
	with open(path_coords1, 'r') as file1, open(path_coords2, 'r') as file2:
		# Jump over the first header.
		file1.readline()
		file2.readline()
		to_terminate = False
		while not to_terminate:
			# Read files.
			file_op = [file1, file2]
			coordinate_list = [[], []]
			pids = [[], []]
			for i in range(2):
				while True:
					line = file_op[i].readline()
					if line == '':
						to_terminate = True
						break
					elif line[0] == '#':
						break
					splits = line.split()
					coordinate_list[i].append([float(splits[0]), float(splits[1])])
					pids[i].append(int(splits[2]))

			coordinates1 = np.array(coordinate_list[0])
			coordinates2 = np.array(coordinate_list[1])
			pids1 = pids[0]
			pids2 = pids[1]

			j = len(pids2)
			for k in range(len(pids2)):
				if pids2[k] > 100:
					j = k
					break
			coords.append(np.hstack([coordinates1[:j, :], coordinates2[:j, :]]))

		coordinates = np.vstack(coords)
		np.savetxt('dataset/coords_w.txt', coordinates)


def show_calib_error():
	coords = np.loadtxt('dataset/coords_w.txt')
	X1 = coords[:, 0]
	Y1 = coords[:, 1]
	X2 = coords[:, 2]
	Y2 = coords[:, 3]
	plt.scatter(X1, Y1, s=10)
	plt.scatter(X2, Y2, s=10)
	plt.axis('equal')
	plt.show()


def detect_and_tracking():

	font = cv2.FONT_HERSHEY_DUPLEX
	font_scale = 0.8

	DIR = 'C:/E/Matlab/Object Tracking/dataset/cam2'

	frame = cv2.imread(os.path.join(DIR, 'img', '0013.jpg'))

	with open(os.path.join(DIR, 'track.txt')) as track, open(os.path.join(DIR, 'detect.txt')) as detect:
		while track.readline() != '#frame\t13\n':
			pass
		while detect.readline() != '#frame 13\n':
			pass

		ntrack = int(track.readline().split()[0])
		for i in range(ntrack):
			line = track.readline()
			splits = line.split()
			pid, x, y, w, h = [int(s) for s in splits]
			cv2.rectangle(frame, (x, y), (x+w, y+h), (59, 235, 255), 2)
			cv2.putText(frame, str(pid), (x, y), font, font_scale, (59, 235, 255), 2)

		ndetect = int(detect.readline().split()[0])
		for i in range(ndetect):
			line = detect.readline()
			splits = line.split()
			x, y, x2, y2 = [int(s) for s in splits]
			cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

	cv2.namedWindow('img', 0)
	cv2.imshow('img', frame)
	cv2.waitKey(0)

if __name__ == '__main__':
	DIR = 'C:/E/Matlab/Object Tracking/dataset/cam1'

	# TOP = 'C:/E/Python/Tracking/dataset'
	# features1 = np.loadtxt(os.path.join(TOP, 'features1.txt'))
	# features2 = np.loadtxt(os.path.join(TOP, 'features2.txt'))
	# with open(os.path.join(TOP, 'images1.txt'), 'r') as file:
	# 	images1 = file.readlines()
	# with open(os.path.join(TOP, 'images2.txt'), 'r') as file:
	# 	images2 = file.readlines()

	# min_distance_hist(features1, images1, features2, images2, intra_camera=True)
	# foo('dataset/positions1.txt', 'dataset/positions2.txt')
	# coords_w = np.loadtxt('dataset/coords_w.txt')
	# print(coords_w)

	detect_and_tracking()