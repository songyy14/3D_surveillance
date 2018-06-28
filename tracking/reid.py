import os
import copy
import sys
sys.path.append('trinet')

import cv2
import numpy as np
# from trinet.inference import euclidean_dist, FeatureExtractor
from trinet.inference import *

# Frame rate of videos.
fps = 25

# Distance threshold used to determine whether to accept
# the rank-1 match.
# distance_th = 15.10
distance_th = 16

# Ratio between minimum distance in intra-camera queries and
# that in extra-camera queries.
ratio = 1.16

# If a person is not matched for this period of time, it is removed
# from gallery. (Measured in seconds)
max_living_time_s = 20


class Gallery:
	'''
	A class for managing database (often called `gallery` in person re-id cases) and
	searching persons in it.
	'''
	def __init__(self):
		self.features = np.zeros((0, 128))
		# ids for persons, cameras and frames, respectively.
		self.pids = []
		self.cids = []
		self.fids = []

	def query(self, features_p, cid_p, fid_p, pids_this_cam, blacklist):
		'''
		query a batch of probes images captured in one camera in one frame.
		
		Args:
			features_p: Numpy array of shape (m, 128).
			cid_p: Id of the camera capturing the images.
			fid_p: Id of the frame.
			pids_this_cam: A list of ids of persons currently tracked by the
				camera, which are excluded in the search for all probes.
			blacklist: Nested list of depth 2, indicating which pids besides
				`pids_this_cam` are impossible to match for each probe.

		Returns:
			queried_pids: The queried person ids for the probe images.
		'''
		# Clean up persons running out of living time.
		i = 0
		while i < len(self.pids):
			if fid_p - self.fids[i] > fps * max_living_time_s:
				print('---remove person %d from gallery' % self.pids[i])
				self.features = np.delete(self.features, i, axis=0)
				del self.pids[i]
				del self.cids[i]
				del self.fids[i]
			else:
				i = i + 1

		# Exclude `pids_this_cam` from the search.
		candidate_pids = []
		for pid in self.pids:
			if pid not in pids_this_cam:
				candidate_pids.append(pid)

		# The candidate list is not empty.
		if len(candidate_pids) > 0:
			# Rows of the records corresponding to the candidates.
			rows = []
			for i in candidate_pids:
				rows.append(self.pids.index(i))

			# Calculate distance matrix and scale the intra-camera distance by `ratio`.
			sub_features = self.features[rows, :]
			dist_matrix = euclidean_dist(features_p, sub_features)
			for j in range(len(rows)):
				if self.cids[rows[j]] == cid_p:
					dist_matrix[:, j] = dist_matrix[:, j] * ratio

			# Penalize the matches in `blacklist` by raising its distance above `distance_th`.
			for i in range(features_p.shape[0]):
				for j in range(len(candidate_pids)):
					if candidate_pids[j] in blacklist[i]:
						dist_matrix[i, j] = dist_matrix[i, j] + 100

			indices = np.argsort(dist_matrix, axis=1)
			idxs = solve_conflicts(indices, dist_matrix)
			new_pid = max(self.pids)

			# Print the sorting result.
			indices_pids = np.zeros(indices.shape, dtype=np.int32)
			for i in range(indices.shape[0]):
				for j in range(indices.shape[1]):
					indices_pids[i, j] = candidate_pids[indices[i, j]]

			print(indices_pids)
			print(np.sort(dist_matrix, axis=1))
		
		# The candidate list is empty, so all probe images are added to gallery.
		else:
			idxs = [-1 for i in range(features_p.shape[0])]
			if len(self.pids) > 0:
				new_pid = max(self.pids)
			else:
				new_pid = 0

		queried_pids = []
		new_features = [self.features]
		for i in range(len(idxs)):
			if idxs[i] >= 0:
				queried_pids.append(candidate_pids[idxs[i]])
				# Update feature and corresponding info for matched persons.
				self.features[rows[idxs[i]], :] = features_p[i, :]
				self.cids[rows[idxs[i]]] = cid_p
				self.fids[rows[idxs[i]]] = fid_p
			else:
				new_pid = new_pid + 1
				queried_pids.append(new_pid)
				# Add this person to the gallery.
				new_features.append(np.expand_dims(features_p[i, :], axis=0))
				self.pids.append(new_pid)
				self.cids.append(cid_p)
				self.fids.append(fid_p)

		# Add new features all in once.
		self.features = np.vstack(new_features)

		return queried_pids

	def get_pids(self):
		return copy.deepcopy(self.pids)

def solve_conflicts(indices, dist_matrix):
	'''
	Solve the conflicts in the rank-1 matches of a batch of queries.

	Args:
		indices: Index array generated when row-wise sorting `dist_matrix` 
			in ascent order.
		dist_matrix: Numpy array, representing the distance between each
			image from query set and each image from database.

	Returns:
		idxs: Indices of matched database image for each query after solving
			conflicts. If the index is below zero, this person is predicted
			to be outside the database.
	'''
	# Initialization.
	m, n = indices.shape
	count = 0
	cols = [0 for i in range(m)]
	idxs = list(indices[:, 0])

	# Take those rank-1 matches whose distance are above threshold out of 
	# consideration.
	for i in range(m):
		if dist_matrix[i, idxs[i]] >= distance_th:
			count = count + 1
			idxs[i] = - count

	while True:
		# Find one pair of matches with conflict.
		r2 = -1
		for r1 in range(m):
			if idxs[r1] in idxs[r1+1:]:
				r2 = idxs.index(idxs[r1], r1 + 1)
				break
		# No conflict anymore.
		if r2 == -1:
			break

		# Solve the conflict:
		# One of the two matches with shorter distance remains unchanged,
		# while the other is moved to the less-optimal match.
		rm = r2
		if dist_matrix[r1, idxs[r1]] > dist_matrix[r2, idxs[r2]]:
			rm = r1
		cols[rm] = cols[rm] + 1
		if cols[rm] > n - 1:
			count = count + 1
			idxs[rm] = - count
		else:
			idxs[rm] = indices[rm, cols[rm]]
			if dist_matrix[rm, idxs[rm]] >= distance_th:
				count = count + 1
				idxs[rm] = - count

	return idxs