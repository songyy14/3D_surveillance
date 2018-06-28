import os

import cv2
import numpy as np
import tensorflow as tf

from nets.resnet_v1 import resnet_v1_50, resnet_arg_scope
from heads.fc1024 import head

slim = tf.contrib.slim

# Some constants.
net_input_height = 256
net_input_width = 128
embedding_dim = 128
_RGB_MEAN = [123.68, 116.78, 103.94]


class FeatureExtractor:
	'''
	A class for extracting features of persons using trinet.
	'''
	def __init__(self, checkpoint):
		self.input = tf.placeholder(dtype=tf.float32, shape=(net_input_height, net_input_width, 3))
		self.output = network_entire(tf.expand_dims(self.input, 0))
		self.sess = tf.Session()
		tf.train.Saver().restore(self.sess, checkpoint)

	def feature(self, image):
		f = self.sess.run(self.output, feed_dict={self.input: image})
		return f

	def close(self):
		self.sess.close()


def adjust_image(image):
	'''
	Adjust person image to fit tensorflow model input.
	'''
	image_f = np.float32(image)
	image_f = np.flip(image_f, axis=2)
	image_f = cv2.resize(image_f, (net_input_width, net_input_height))
	return image_f


def euclidean_dist(X, Y):
	'''
	Args:
		X, Y: A matrix, each row of which corresponds to a sample.

	Returns:
		dist: A matrix, whose (i, j) entry respresents euclidean distance between
			the i-th sample from `X` and the j-th sample from Y.
	'''
	# (x-y)(x-y)' = xx'+yy'-2xy'
	x_self_product = np.sum(X * X, axis=1, keepdims=True)
	y_self_product = np.sum(Y * Y, axis=1, keepdims=True)
	cross_product = X @ Y.T

	dist = x_self_product @ np.ones((1, Y.shape[0])) + np.ones((X.shape[0], 1)) @ y_self_product.T - 2 * cross_product
	dist = np.sqrt(dist)

	return dist


def network_entire(images):
	'''
	A tensorflow operation that extracts features for a batch of images.

	Args:
		images: Numpy array of shape (n, h, w, 3).

	Returns:
		embedding: Tensor of shape (n, 128).
	'''
	# Normalization.
	images = images - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))

	# Travel through the network and get the embedding.
	with slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
		_, endpoints = resnet_v1_50(images, num_classes=None, is_training=False, global_pool=True)

	endpoints['model_output'] = endpoints['global_pool'] = tf.reduce_mean(
		endpoints['resnet_v1_50/block4'], [1, 2], name='pool5', keep_dims=False)

	with tf.name_scope('head'):
		endpoints = head(endpoints, embedding_dim, is_training=False)

	embedding = endpoints['emb']

	return embedding


def extract_features(images):
	'''
	Extract features for a batch of images using the trained Resnet_v1_50 model.

	Args:
		images: Numpy array of shape (n, h, w, 3).

	Returns:
		A numpy array of shape (n, 128), each row of which is a feature vector for one image.
	'''
	path_checkpoint = 'C:/E/Python/Tracking/trinet/checkpoint/checkpoint.ckpt-25000'
	embedding = network_entire(images)
	with tf.Session() as sess:
		tf.train.Saver().restore(sess, path_checkpoint)
		features = sess.run(embedding)

	return features


def gen_batch(path_images):
	
	path = tf.placeholder(tf.string)
	image_string = tf.read_file(path)
	image = tf.image.decode_jpeg(image_string, channels=3, dct_method='INTEGER_ACCURATE')
	image = tf.image.resize_images(image, (net_input_height, net_input_width))

	img_list = []
	with tf.Session() as sess:
		for file in path_images:
			img_list.append(sess.run(image, feed_dict={path: file}))

	batch = np.stack(img_list, axis=0)
	return batch


if __name__ == '__main__':
	DATASET = 'C:/E/Matlab/Object Tracking/dataset'
	path_checkpoint = 'C:/E/Python/Tracking/trinet/checkpoint/checkpoint.ckpt-25000'

	# dir1 = os.path.join(DATASET, 'cam1', 'persons')
	# dir2 = os.path.join(DATASET, 'cam1', 'persons_13')
	# dir3 = os.path.join(DATASET, 'cam2', 'persons')
	# paths1 = [os.path.join(dir1, '%d.jpg' % i) for i in range(1, 1+len(os.listdir(dir1)))]
	# paths2 = [os.path.join(dir2, '%d.jpg' % i) for i in range(1, 1+len(os.listdir(dir2)))]
	# paths3 = [os.path.join(dir3, '%d.jpg' % i) for i in range(1, 1+len(os.listdir(dir3)))]

	dir1 = '../dataset/cam1'
	dir2 = '../dataset/cam2'
	files1 = os.listdir(dir1)
	files2 = os.listdir(dir2)

	files_op = [files1, files2]
	dir_op = [dir1, dir2]

	fe = FeatureExtractor(path_checkpoint)
	for i in range(1, 3):
		feature_list = []
		files = files_op[i - 1]
		dir_imgs = dir_op[i - 1]
		for f in files:
			img = cv2.imread(os.path.join(dir_imgs, f))
			img = adjust_image(img)
			feature = fe.feature(img)
			feature_list.append(feature)
		features = np.vstack(feature_list)
		np.savetxt('../dataset/features%d.txt' % i, features)
		with open('../dataset/images%d.txt' % i, 'w') as out:
			for f in files:
				out.write(f + '\n')

	fe.close()