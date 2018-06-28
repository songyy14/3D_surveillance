import os

import cv2
import numpy as np

import KCF
from util import detection_query, read_camera_matrix, get_blacklist
from trinet.inference import FeatureExtractor, adjust_image
from reid import Gallery


time_to_detect = {
	'cam1': 1,
	'cam2': 4,
	'cam3': 7,
	'cam4': 0
}

def get_color(cam, pid, queried_pids):
	'''
	Customize drawing color for each camera.
	'''
	if pid in queried_pids[cam]:
		return (0, 255, 0)

	if cam == 'cam4':
		return (54, 67, 244)
	elif cam == 'cam3':
		return (59, 235, 255)
	elif cam == 'cam2':
		return (74, 195, 139)
	else:
		return (212, 188, 0)


def main():

	# Ouput settings.
	DRAW = True
	SAVE_TO_FILE = False
	SAVE_FRAMES = False

	font = cv2.FONT_HERSHEY_DUPLEX
	font_scale = 0.8
	thickness = 2

	# Tracking settings.
	HOG = True
	FIXEDWINDOW = True
	MULTISCALE = True
	LAB = False
	
	fe = FeatureExtractor('trinet/checkpoint/checkpoint.ckpt-25000')
	gallery = Gallery()

	PATH_TOP = 'C:/E/Matlab/Object Tracking/dataset'
	cams = ['cam2', 'cam3']

	queried_pids = dict()

	trackers = dict()
	outputs = dict()
	fdetects = dict()
	camera_matrices = dict()
	for cam in cams:
		queried_pids[cam] = []
		trackers[cam] = []
		fdetects[cam] = open(os.path.join(PATH_TOP, cam, 'detect.txt'), 'r')
		camera_matrices[cam] = read_camera_matrix(os.path.join(PATH_TOP, cam, 'camera_matrix_our.txt'))
		if DRAW:
			cv2.namedWindow(str(cam), 0)
		if SAVE_TO_FILE:
			outputs[cam] = open(os.path.join(PATH_TOP, cam, 'track23.txt'), 'w')

	start_count = cv2.getCPUTickCount()
	num_frames = len(os.listdir(os.path.join(PATH_TOP, 'cam1', 'img')))

	# Main loop.
	for i in range(1, num_frames + 1):
		print('\n#frame %d' % i)
		for cam in cams:
			print('')
			print('       ', cam)
			print('')
			frame = cv2.imread(os.path.join(PATH_TOP, cam, 'img', '%04d.jpg' % i))

			# Update tracking positions and delete some bad trackers.
			j = 0
			while j < len(trackers[cam]):
				trackers[cam][j].update(frame)
				if trackers[cam][j].out_of_sight:
					print("Delete tracker %d due to out of sight" % trackers[cam][j].pid)
					del trackers[cam][j]
				elif trackers[cam][j].occluded_so_long:
					print("Delete tracker %d due to occluded so long" % trackers[cam][j].pid)
					del trackers[cam][j]
				else:
					j = j + 1

			# Add new trackers every 10 frames, of course including the first frame.
			if i % 10 == time_to_detect[cam]:
				# Note we have not deleted the model-drifted trackers.
				# Sometimes good trackers are considered as model drift,
				# due to the imperfect criterion.
				# So if we delete these trackers and re-add them, the tracking result
				# may look consistent.

				# First, delete model-drifted trackers.
				j = 0
				while j < len(trackers[cam]):
					if trackers[cam][j].model_drift:
						print("Delete tracker %d due to model drift" % trackers[cam][j].pid)
						del trackers[cam][j]
					else:
						j = j + 1

				# Then, add new trackers.
				
				# Read detection results of current frame.
				# Locate current frame.
				while fdetects[cam].readline() != ('#frame %d\n' % i):
					pass
				num_detect = int(fdetects[cam].readline().split()[0])
				detect_pos = np.zeros((num_detect, 4), dtype=np.int32)
				for j in range(num_detect):
					line = fdetects[cam].readline()
					splits = line.split()
					tmp = [int(k) for k in splits]
					# Detection bounding boxes are in the form of (x1, y1, x2, y2).
					detect_pos[j, :] = [tmp[0], tmp[1], tmp[2] - tmp[0], tmp[3] - tmp[1]]

				# Put tracking results together.
				track_pos = np.zeros((len(trackers[cam]), 4))
				for j in range(len(trackers[cam])):
					track_pos[j, :] = trackers[cam][j].get_roi()

				# Determine which detection boxes are used to initialize new trackers.
				indices = detection_query(detect_pos, track_pos)
				
				if len(indices) > 0:
					# Person re-identification.				
					feature_list = []
					blacklist = []
					for j in range(len(indices)):
						x, y, w, h = detect_pos[indices[j], :]
						person_img = frame[y:y+h, x:x+w, :]
						person_feature = fe.feature(adjust_image(person_img))
						feature_list.append(person_feature)
						# Get blacklist for this person.
						one_blacklist = get_blacklist(x + w/2, y + h, cam, trackers, camera_matrices)
						blacklist.append(one_blacklist)

					features_p = np.vstack(feature_list)
					gallery_pids = gallery.get_pids()
					pids_this_cam = [tracker.pid for tracker in trackers[cam]]
					queried_pids[cam] = gallery.query(features_p, cam, i, pids_this_cam, blacklist)

					# Initialize new trackers with the queried pids.
					# frame_b = frame.copy()
					# new_persons = False
					for j in range(len(indices)):
						tracker = KCF.kcftracker(queried_pids[cam][j], HOG, FIXEDWINDOW, MULTISCALE, LAB)
						tracker.init(list(detect_pos[indices[j], :]), frame)
						trackers[cam].append(tracker)
						if queried_pids[cam][j] in gallery_pids:
							print('---------------- Resume tracker %d' % queried_pids[cam][j])
						else:
							print('---------------- Add tracker %d' % queried_pids[cam][j])
							# new_persons = True
						# x, y, w, h = detect_pos[indices[j], :]
						# cv2.rectangle(frame_b, (x, y), (x + w, y + h), get_color(cam, -1, queried_pids), thickness, 8)
						# cv2.putText(frame_b, str(queried_pids[cam][j]), (x, y), font, font_scale, get_color(cam, -1, queried_pids), thickness)

					# if new_persons:
					# cv2.imwrite(os.path.join(PATH_TOP, 'cam1', 'gallery', '%d.jpg' % i), frame_b)


			# Draw and save trackers positions to file.
			if SAVE_TO_FILE:
				outputs[cam].write('#frame\t%d\n' % i)
				outputs[cam].write('%d\n' % len(trackers[cam]))

			for j in range(len(trackers[cam])):
				x, y, w, h = trackers[cam][j].get_roi()
				if SAVE_TO_FILE:
					outputs[cam].write('%d\t%d\t%d\t%d\t%d\n' % (trackers[cam][j].pid, x, y, w, h))

				if DRAW:
					cv2.rectangle(frame, (x, y), (x + w, y + h), get_color(cam, trackers[cam][j].pid, queried_pids), thickness, 8)
					cv2.putText(frame, str(trackers[cam][j].pid), (x, y), font, font_scale, get_color(cam, trackers[cam][j].pid, queried_pids), thickness)

			if DRAW:
				cv2.imshow(str(cam), frame)
				cv2.waitKey(0)
				if i == num_frames:
					cv2.waitKey(0)

			if SAVE_FRAMES:
				# Save frames with tracking results.
				cv2.imwrite(os.path.join(PATH_TOP, cam, 'img_tracking_reid', '%04d.jpg' % i), frame)

	# Release resources.
	fe.close()
	for cam in cams:
		fdetects[cam].close()
	if SAVE_TO_FILE:
		for cam in cams:
			outputs[cam].close()

	elapsed_time_s = float(cv2.getCPUTickCount() - start_count) / cv2.getTickFrequency()
	fps = num_frames / elapsed_time_s
	print('%f fps' % fps)



if __name__ == '__main__':
	main()