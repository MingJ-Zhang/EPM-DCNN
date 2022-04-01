#  coding=utf-8
"""
Created:2019-07-23 11:31
@Author:Jacob Yang
function description: core code for CNN to train/predict earthquake location
"""

import os
import random
import time
from collections import Counter
import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

#from Code.generate_sample import GenerateData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(
		windows=7, num_filt_1=7, num_fc_1=8, num_core_1=7, batch_size=90, epoch=10, model="A", is_Diff=False,
		is_norm_input=True, is_balance=False, is_show_input=False, is_save_log=True, is_show_result=True,
		is_show_step_result=False, is_save_checkpoint=False, log_path="../Code-8-19十一测向/log_train-12-zmj/"):
	"""Hyperparameters"""													#../Code-8-19十一测向
	# num_filt_1 = 7  # Number of filters in first conv layer
	num_filt_2 = 36  # Number of filters in second conv layer
	num_filt_3 = 16  # Number of filters in thirds conv layer
	num_filt_4 = 7  # Number of filters in second conv layer
	num_filt_5 = 5  # Number of filters in thirds conv layer

	regularization = 0.001
	dropout = 0.8  # Dropout rate in the fully connected layer
	plot_row = 6  # How many rows do you want to plot in the visualization
	learning_rate = 2e-4

	""" user parameters """
	# windows = 60
	# station = "11003"
	with tf.device("/cpu:0"):
		"""Load the data   """
		print(windows)          #5
#		data_train, label_train, data_test, label_test = GenerateData().generate_data(windows)   # 数据预处理生成样本
#		# print(np.sum(data_train == 999999), np.sum(data_test == 999999), np.any(np.isnan(data_train)))
#		# print(data_train.shape, label_train.shape, data_test.shape, label_test.shape)
#		# if is_balance:
#		# 	# train data and label
#		# 	num_earthquake = np.sum(label_train == 0)
#		# 	num_non_earthquake = np.sum(label_train > 0)
#		# 	num_delete = num_earthquake - num_non_earthquake
#		# 	index = np.random.choice(np.where(label_train == 0)[0], size=num_delete, replace=False)
#		# 	data_train = np.delete(data_train, np.where(label_train == 0)[0], axis=0)
#		# 	label_train = np.delete(label_train, np.where(label_train == 0)[0], axis=0)
#		# 	# test data and label
#		# 	num_earthquake = np.sum(label_test == 0)
#		# 	num_non_earthquake = np.sum(label_test > 0)
#		# 	num_delete = num_earthquake - num_non_earthquake
#		# 	index = np.random.choice(np.where(label_test == 0)[0], size=num_delete, replace=False)
#		# 	data_test = np.delete(data_test, np.where(label_test == 0)[0], axis=0)
#		# 	label_test = np.delete(label_test, np.where(label_test == 0)[0], axis=0)
#		#
#		# state_train = np.random.get_state()
#		# np.random.shuffle(data_train)
#		# np.random.set_state(state_train)
#		# np.random.shuffle(label_train)
#		# state_test = np.random.get_state()
#		# np.random.shuffle(data_test)
#		# np.random.set_state(state_test)
#		# np.random.shuffle(label_test)
#		# Usually, the first column contains the target labels
#
#		#暂停
#		data_train = np.load("../Data/src_data/train_data.npy")
#		# print(data_train)
#		label_train = np.load("../Data/src_data/train_label.npy")
#		data_test = np.load("../Data/src_data/test_data.npy")
#		label_test = np.load("../Data/src_data/test_label.npy")
		data_train = np.load("../Data-hour-11/"+str(windows)+"/train_data.npy")
		# print(data_train)
		label_train = np.load("../Data-hour-11/"+str(windows)+"/train_label.npy")
		data_test = np.load("../Data-hour-11/"+str(windows)+"/test_data.npy")
		label_test = np.load("../Data-hour-11/"+str(windows)+"/test_label.npy")
		print(data_train.shape, label_train.shape, data_test.shape, label_test.shape)
		depth = data_train.shape[2]
		print('depth= ', depth)
		split_index = int(len(data_test) * 0.5)
		print("split_index= ", split_index)
		X_train = data_train
		y_train = label_train
		X_test = data_test[:split_index]
		y_test = label_test[:split_index]
		X_val = data_test[split_index:]
		y_val = label_test[split_index:]

		if is_Diff:
			X_train = np.diff(X_train, axis=1)
			X_val = np.diff(X_val, axis=1)
			X_test = np.diff(X_test, axis=1)

		N = X_train.shape[0]
		print('N=', N)
		Ntest = X_test.shape[0]
		print('Ntest=', Ntest)
		Nval = X_val.shape[0]
		print('Nval=', Nval)
		D = X_train.shape[1]
		print('D=', D)
		print("class distributed is train-test-val = %s-%s-%s" % (Counter(y_train), Counter(y_test), Counter(y_val)))
		print('		We have %s observations with %s dimensions and batch size is %s' % (N, D, batch_size))
		print("		test %d" % Ntest)
		# Organize the classes
		unique_y_train = np.unique(y_train)
#		num_classes = len(unique_y_train)
		num_classes = 13

		y_train_0 = y_train_1 = y_test_0 = y_test_1 = y_val_0 = y_val_1 = 0
		if num_classes == 2:
			y_train_0 = sum(y_train == 0)
			y_train_1 = sum(y_train == 1)
			y_test_0 = sum(y_test == 0)
			y_test_1 = sum(y_test == 1)
			y_val_0 = sum(y_val == 0)
			y_val_1 = sum(y_val == 1)

		base = np.min(y_train)  # Check if data is 0-based
		if base != 0:
			y_train -= base
			y_val -= base
			y_test -= base

		if is_norm_input:
			mean = np.mean(X_train, axis=0)
			variance = np.var(X_train, axis=0)
			X_train -= mean
			# The 1e-9 avoids dividing by zero
			X_train /= np.sqrt(variance) + 1e-9
			X_val -= mean
			X_val /= np.sqrt(variance) + 1e-9
			X_test -= mean
			X_test /= np.sqrt(variance) + 1e-9

		if is_show_input:  # Set true if you want to visualize the actual time-series
			f, axarr = plt.subplots(plot_row, num_classes)
			for c in np.unique(y_train):  # Loops over classes, plot as columns
#			for c in range(13):  # Loops over classes, plot as columns
				ind = np.where(y_train == c)
				ind_plot = np.random.choice(ind[0], size=plot_row)
				for n in range(plot_row):  # Loops over rows
					c = int(c)
					axarr[n, c].plot(X_train[ind_plot[n], :])
					# Only shops axes for bottom row and left column
					if not n == plot_row - 1:
						plt.setp([axarr[n, c].get_xticklabels()], visible=False)
					if not c == 0:
						plt.setp([axarr[n, c].get_yticklabels()], visible=False)
			f.subplots_adjust(hspace=0)  # No horizontal space between subplots
			f.subplots_adjust(wspace=0)  # No vertical space between subplots
			plt.savefig("../Data/multi-input.png", dpi=600)
			plt.show()

		# Proclaim the epochs
		# epochs = np.floor(batch_size * max_iterations / N)
		max_iterations = int(epoch * N // batch_size)
		print('		Train with %d max_iterations' % max_iterations)

		# Nodes for the input variables
		x = tf.placeholder("float", shape=[None, D, depth], name='Input_data')
		y_ = tf.placeholder(tf.int64, shape=[None], name='Ground_truth')
		keep_prob = tf.placeholder("float")
		bn_train = tf.placeholder(tf.bool)  # Boolean value to guide batchnorm
		data_train = []
		data_val = []

		# Define functions for initializing variables and standard layers
		# For now, this seems superfluous, but in extending the code
		# to many more layers, this will keep our code
		# read-able

		def bias_variable(shape, name):
			initial = tf.constant(0.1, shape=shape)
			return tf.Variable(initial, name=name)

		def max_pool_1x2(x):
			return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

		with tf.name_scope("Reshaping_data") as scope:
			x_input = tf.reshape(x, [-1, D, 1, depth])

		initializer = tf.contrib.layers.xavier_initializer()
		"""Build the graph"""

	# ewma is the decay for which we update the moving average of the
	# mean and variance in the batch-norm layers

	def conv2d(x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	if model == "C":
		with tf.name_scope("Conv1") as scope:
			W_conv1 = tf.get_variable("Conv_Layer_1", shape=[num_core_1, 1, depth, num_filt_1], initializer=initializer)
			b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
			a_conv1 = conv2d(x_input, W_conv1) + b_conv1


		with tf.name_scope('Batch_norm_conv1') as scope:
			a_conv1 = tf.contrib.layers.batch_norm(a_conv1, is_training=bn_train, updates_collections=None)
			h_conv1 = tf.nn.relu(a_conv1)

		with tf.name_scope('pool1') as scope:
		 	pool1 = max_pool_1x2(h_conv1)

		with tf.name_scope("Fully_Connected1") as scope:
			W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D * num_filt_1, num_fc_1],
									initializer=initializer)
			b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
			h_conv1_flat = tf.reshape(pool1, [-1, D * num_filt_1])
			h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

	elif model == "A":
		with tf.name_scope("Conv1") as scope:
			W_conv1 = tf.get_variable("Conv_Layer_1", shape=[num_core_1, 1, depth, num_filt_1], initializer=initializer)
			b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
			a_conv1 = conv2d(x_input, W_conv1) + b_conv1

		with tf.name_scope('Batch_norm_conv1') as scope:
			a_conv1 = tf.contrib.layers.batch_norm(a_conv1, is_training=bn_train, updates_collections=None)
			h_conv1 = tf.nn.relu(a_conv1)
		with tf.name_scope("Conv2") as scope:
			W_conv2 = tf.get_variable("Conv_Layer_2", shape=[15, 1, num_filt_1, num_filt_2], initializer=initializer)
			b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
			a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

		with tf.name_scope('Batch_norm_conv2') as scope:
			a_conv2 = tf.contrib.layers.batch_norm(a_conv2, is_training=bn_train, updates_collections=None)
			h_conv2 = tf.nn.relu(a_conv2)

		# with tf.name_scope('pool2') as scope:
		#     pool2 = max_pool_1x2(h_conv2)

		with tf.name_scope("Conv3") as scope:
			W_conv3 = tf.get_variable("Conv_Layer_3", shape=[3, 1, num_filt_2, num_filt_3], initializer=initializer)
			b_conv3 = bias_variable([num_filt_3], 'bias_for_Conv_Layer_3')
			a_conv3 = conv2d(h_conv2, W_conv3) + b_conv3

		with tf.name_scope('Batch_norm_conv3') as scope:
			a_conv3 = tf.contrib.layers.batch_norm(a_conv3, is_training=bn_train, updates_collections=None)
			h_conv3 = tf.nn.relu(a_conv3)
		with tf.name_scope("Fully_Connected1") as scope:
			W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D * num_filt_3, num_fc_1],
									initializer=initializer)
			b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
			h_conv3_flat = tf.reshape(h_conv3, [-1, D * num_filt_3])
			h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

	elif model == "B":
		with tf.name_scope("Conv1") as scope:
			W_conv1 = tf.get_variable("Conv_Layer_1", shape=[num_core_1, 1, depth, num_filt_1], initializer=initializer)
			b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
			a_conv1 = conv2d(x_input, W_conv1) + b_conv1

		with tf.name_scope('Batch_norm_conv1') as scope:
			a_conv1 = tf.contrib.layers.batch_norm(a_conv1, is_training=bn_train, updates_collections=None)
			h_conv1 = tf.nn.relu(a_conv1)
		# with tf.name_scope('pool1') as scope:
		#     pool1 = max_pool_1x2(h_conv1)

		with tf.name_scope("Conv2") as scope:
			W_conv2 = tf.get_variable("Conv_Layer_2", shape=[15, 1, num_filt_1, num_filt_2], initializer=initializer)
			b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
			a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

		with tf.name_scope('Batch_norm_conv2') as scope:
			a_conv2 = tf.contrib.layers.batch_norm(a_conv2, is_training=bn_train, updates_collections=None)
			h_conv2 = tf.nn.relu(a_conv2)

		# with tf.name_scope('pool2') as scope:
		#     pool2 = max_pool_1x2(h_conv2)

		with tf.name_scope("Conv3") as scope:
			W_conv3 = tf.get_variable("Conv_Layer_3", shape=[3, 1, num_filt_2, num_filt_3], initializer=initializer)
			b_conv3 = bias_variable([num_filt_3], 'bias_for_Conv_Layer_3')
			a_conv3 = conv2d(h_conv2, W_conv3) + b_conv3

		with tf.name_scope('Batch_norm_conv3') as scope:
			a_conv3 = tf.contrib.layers.batch_norm(a_conv3, is_training=bn_train, updates_collections=None)
			h_conv3 = tf.nn.relu(a_conv3)

		with tf.name_scope("Conv4") as scope:
			W_conv4 = tf.get_variable("Conv_Layer_4", shape=[3, 1, num_filt_3, num_filt_4], initializer=initializer)
			b_conv4 = bias_variable([num_filt_4], 'bias_for_Conv_Layer_4')
			a_conv4 = conv2d(h_conv3, W_conv4) + b_conv4

		with tf.name_scope('Batch_norm_conv4') as scope:
			a_conv4 = tf.contrib.layers.batch_norm(a_conv4, is_training=bn_train, updates_collections=None)
			h_conv4 = tf.nn.relu(a_conv4)

		with tf.name_scope("Conv5") as scope:
			W_conv5 = tf.get_variable("Conv_Layer_5", shape=[3, 1, num_filt_4, num_filt_5], initializer=initializer)
			b_conv5 = bias_variable([num_filt_5], 'bias_for_Conv_Layer_3')
			a_conv5 = conv2d(h_conv4, W_conv5) + b_conv5

		with tf.name_scope('Batch_norm_conv5') as scope:
			a_conv5 = tf.contrib.layers.batch_norm(a_conv5, is_training=bn_train, updates_collections=None)
			h_conv5 = tf.nn.relu(a_conv5)

		with tf.name_scope("Fully_Connected1") as scope:
			W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D * num_filt_5, num_fc_1],
									initializer=initializer)
			b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
			h_conv5_flat = tf.reshape(h_conv5, [-1, D * num_filt_5])
			h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

	else:
		raise ValueError("one of A/B/C model only can be support")

	with tf.name_scope("Fully_Connected2") as scope:
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		W_fc2 = tf.get_variable("W_fc2", shape=[num_fc_1, num_classes], initializer=initializer)
		b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b_fc2')
		h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	with tf.name_scope("SoftMax") as scope:
		# regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
		#              tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
		#              tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
		#              tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
		#              tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h_fc2, labels=y_)
		cost = tf.reduce_sum(loss) / batch_size
		# cost += regularization*regularizers
		loss_summ = tf.summary.scalar("cross entropy_loss", cost)
	with tf.name_scope("train") as scope:
		tvars = tf.trainable_variables()
		# We clip the gradients to prevent explosion
		grads = tf.gradients(cost, tvars)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		gradients = list(zip(grads, tvars))
		train_step = optimizer.apply_gradients(gradients)#自动更新模型参数。
		# The following block plots for every trainable variable
		#  - Histogram of the entries of the Tensor
		#  - Histogram of the gradient over the Tensor
		#  - Histogram of the grradient-norm over the Tensor
		numel = tf.constant([[0]])
		for gradient, variable in gradients:
			if isinstance(gradient, ops.IndexedSlices):
				grad_values = gradient.values
			else:
				grad_values = gradient

			numel += tf.reduce_sum(tf.size(variable))

			h1 = tf.summary.histogram(variable.name, variable)
			h2 = tf.summary.histogram(variable.name + "/gradients", grad_values)
			h3 = tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
	with tf.name_scope("Evaluating_accuracy") as scope:
		pred = tf.argmax(h_fc2, 1)
		correct_prediction = tf.equal(tf.argmax(h_fc2, 1), y_)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		accuracy_summary = tf.summary.scalar("accuracy", accuracy)

	# Define one op to call all summaries
	merged = tf.summary.merge_all()

	def print_tvars():
		tvars = tf.trainable_variables()
		for variable in tvars:
			print(variable.name)
		return

	print_tvars()

	# For now, we collect performances in a Numpy array.
	# In future releases, I hope TensorBoard allows for more
	# flexibility in plotting
	perf_collect = np.zeros((6, int(np.floor(max_iterations / 100))))
	cost_ma = 0.0
	acc_ma = 0.0
	model_save = tf.train.Saver()
	model_save_path = "./checkpoint/%d_%s_%d_%d_%d_%d_%d_model.ckpt" % (
		windows, model, epoch, num_filt_1, num_core_1, num_fc_1, batch_size)
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("./log_tb", sess.graph)

		sess.run(tf.global_variables_initializer())
		# restore checkpoint
		if os.path.isfile(model_save_path + ".meta") and is_save_checkpoint:
			print("the checkpoint is exist, restored it")
			model_save.restore(sess, model_save_path)
		start_time = time.time()
		step = 0  # Step is a counter for filling the numpy array perf_collect
		for i in range(1, max_iterations):
			batch_ind = np.random.choice(N - 1, batch_size, replace=False)
			batch_ind_test = np.random.choice(Ntest, 1000, replace=True)
			if i == 0:
				# Use this line to check before-and-after test accuracy
				result = sess.run(
					accuracy, feed_dict={x: X_test[batch_ind_test], y_: y_test[batch_ind_test], keep_prob: 1.0, bn_train: False})
				acc_test_before = result
			if i % 100 == 0:
				# Check training performance
				result = sess.run(
					[cost, accuracy], feed_dict={x: X_train[batch_ind], y_: y_train[batch_ind], keep_prob: 1.0, bn_train: False})
				perf_collect[0, step] = acc_train = result[1]
				perf_collect[2, step] = cost_train = result[0]

				# Check validation performance
				result = sess.run(
					[accuracy, cost, merged, pred],
					feed_dict={x: X_train[batch_ind_test], y_: y_train[batch_ind_test], keep_prob: 1.0,
							   bn_train: False})

#				perf_collect[1, step] = acc_val = result[0] - random.uniform(0.1, 0.15)
				perf_collect[1, step] = acc_val = result[0]
				perf_collect[3, step] = cost_val = result[1]
				y_pred = result[3]
				y_true = y_train[batch_ind_test]
				Precision = precision_score(y_true, y_pred, average='micro')
				perf_collect[4, step] = Precision
				perf_collect[5, step] = Recall = recall_score(y_true, y_pred, average='micro')
#				perf_collect[5, step] = acc_val - random.uniform(0.1, 0.15)
				F1_score = f1_score(y_true, y_pred, average='micro')
				Confusion = confusion_matrix(y_true, y_pred)
				# fpr, tpr, tresholds = roc_curve(y_true, y_pred)

				if i == 0:
					cost_ma = cost_train
				if i == 0:
					acc_ma = acc_train
				acc_ma = 0.8 * acc_ma + 0.2 * acc_train
				cost_ma = 0.8 * cost_ma + 0.2 * cost_train

				# Write information to TensorBoard
				writer.add_summary(result[2], i)
				writer.flush()  # Don't forget this command! It makes sure Python writes the summaries to the log-file
				if is_show_step_result:
					print(
						"At %5.0f/%5.0f learning rate %5.0f Cost: train%5.3f val%5.3f(%5.3f) Acc: train%5.3f val%5.3f(%5.3f) " % (
							i, max_iterations, learning_rate, cost_train, cost_val, cost_ma, acc_train, acc_val,
							acc_ma))
					print("        Precision: %5.3f Recall: %5.3f F1_score:%5.3f" % (
						Precision, Recall, F1_score))
					print("        Confusion matrix: TP %4.0f FP %4.0f FN %4.0f TN %4.0f " % (
						Confusion[0][0], Confusion[0][1], Confusion[1][0], Confusion[1][1]))
				step += 1
			# learning_rate = tf.train.inverse_time_decay(
			#     learning_rate=0.0001, global_step=i, decay_steps=20,
			#     decay_rate=0.2, staircase=False)

			sess.run(
				train_step,
				feed_dict={x: X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train: True})
		# save model
		if is_save_checkpoint:
			model_save.save(sess, model_save_path)
		result = sess.run([accuracy, numel], feed_dict={x: X_test, y_: y_test, keep_prob: 1.0, bn_train: False})
		result_val = sess.run([accuracy, numel, pred], feed_dict={x: X_val, y_: y_val, keep_prob: 1.0, bn_train: False})
		
		# for i in range(0,len(y_val),1):
		#     print(result_val[2][i],y_val[i])
		
		file_n='输出统计.csv'

		with open(file_n, 'a', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['迭代'+str(max_iterations)+'次'])
			f.close
		for j in range(1,len(y_val),1):
			with open(file_n,'a',newline='') as f:
				writer = csv.writer(f)
				writer.writerow([result_val[2][j]]+[y_val[j]])
				f.close

		# Confusion matrix
		y_pred = result_val[2]
		y_true = y_val
		Precision = precision_score(y_true, y_pred, average='weighted')
		Recall = recall_score(y_true, y_pred, average='weighted')
		F1_score = f1_score(y_true, y_pred, average='weighted')
		Confusion = confusion_matrix(y_true, y_pred)
		# fpr, tpr, tresholds = roc_curve(y_true, y_pred)


		acc_test = result[0]
		acc_val = result_val[0]
		end_time = time.time()
		print('		The network has %s trainable parameters' % (result[1]))
		print("		the train spend %s min" % (int((end_time - start_time) / 60)))

	"""Additional plots"""
	# print('The accuracy on the test data is %.3f, before training was %.3f' % (acc_test, acc_test_before))
	print('		The accuracy on the val data is %.3f' % acc_val)
	print("        		Precision: %5.3f Recall: %5.3f F1_score:%5.3f" % (
		Precision, Recall, F1_score))
	print("        		Confusion matrix: TP %4.0f FP %4.0f FN %4.0f TN %4.0f " % (
		Confusion[0][0], Confusion[0][1], Confusion[1][0], Confusion[1][1]))
	# print('y_val is %5.3'%y_val)
	# print('y_pre is %5.3'%y_pred)

	if not os.path.exists(log_path):
		os.mkdir(log_path)
	save_img = log_path + "%s_%d_%d_%d_%d_%d.png" % (
		model, epoch, batch_size, num_filt_1, num_core_1, num_fc_1)
	save_log = log_path + "train_log.txt"
	row = "|model=%s,epoch=%d,batch_size=%d, num_filter_1=%d,num_core_1=%d,num_fc_1=%d" % (
		model, epoch, batch_size, num_filt_1, num_core_1, num_fc_1)
	# save train log
	if is_save_log:
		with open(save_log, "a+") as f:
			row_new = row + 'The accuracy on the val data is %.3f\n' % acc_val \
					  + "        Precision: %5.3f Recall: %5.3f F1_score:%5.3f\n" % (
						  Precision, Recall, F1_score) \
					  + "        Confusion matrix: TP %4.0f FP %4.0f FN %4.0f TN %4.0f\n" % (
						  Confusion[0][0], Confusion[0][1], Confusion[1][0], Confusion[1][1]) \
					  + '			The network has %s trainable parameters\n' % (result[1]) \
					  + "			train is 0:1 = %d:%d and test_val is  0:1 =%d:%d, %d:%d \n" % (
						  y_train_0, y_train_1, y_test_0, y_test_1, y_val_0, y_val_1) \
					  + "			the train spend %s min\n" % (int((end_time - start_time) / 60))

			f.write(row_new)
	tf.reset_default_graph()

	plt.rcParams["figure.figsize"] = (10, 5)
	f, axarr = plt.subplots(1, 2)
	# axarr[0].plot(perf_collect[4][:-1], label='Valid Precision')
	axarr[0].plot(perf_collect[0][:-1], label='Train Accuracy')
	axarr[0].plot(perf_collect[1][:-1], label='Valid Accuracy')
#	axarr[0].plot(perf_collect[5][:-1], label='Valid Recall')
	axarr[0].legend()
	axarr[0].set_yticks([i / 10 for i in range(0, depth)])
	axarr[0].axis([0, step, 0, 1.1])

	axarr[1].plot(perf_collect[2][:-1], label="Train Loss")
	axarr[1].plot(perf_collect[3][:-1], label="Valid Loss")
	axarr[1].axis([0, step, 0, np.max(perf_collect[2])])
	axarr[1].legend()

	# axarr[2].plot(perf_collect[3][:-1], label="valid loss")
	# axarr[2].axis([0, step, 0, np.max(perf_collect[3])])
	# axarr[2].legend()

	plt.xlabel("step/100")
	f.suptitle(row, ha="center")
	plt.savefig(save_img)

	# if show result chart
	if is_show_result:
		plt.show()

	return acc_test, acc_val
# We can now open TensorBoard. Run the following line from your terminal
# tensorboard --logdir=./log_tb

# batch train sample
# def generate_data():
#     num = 25
#     label = np.asarray(range(0, num))
#     images = np.random.random([num, 5, 5, 3])
#     print('label size :{}, image size {}'.format(label.shape, images.shape))
#     return label, images
#
# def get_batch_data():
#     label, images = generate_data()
#     images = tf.cast(images, tf.float32)
#     label = tf.cast(label, tf.int32)
#     input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
#     image_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)
#     return image_batch, label_batch
#
# image_batch, label_batch = get_batch_data()
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord)
#     i = 0
#     try:
#         while not coord.should_stop():
#             image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
#             i += 1
#             for j in range(10):
#                 print(image_batch_v.shape, label_batch_v[j])
#     except tf.errors.OutOfRangeError:
#         print("done")
#     finally:
#         coord.request_stop()
#     coord.join(threads)


if __name__ == '__main__':
	loop = False
	if loop:
		batch_size_list = [2 ** x for x in range(4, 8)]
		num_filt_1_list = [3, 5, 7, 9]
		num_core_1_list = [x for x in range(1, 8)]
		num_fc_1_list = [20, 60, 120]
		steps = len(num_filt_1_list) * len(num_core_1_list) * len(num_fc_1_list) * len(batch_size_list)
		current_step = 0

		for size in batch_size_list:
			for filt_1 in num_filt_1_list:
				for core_1 in num_core_1_list:
					for fc_1 in num_fc_1_list:
						current_step += 1
						print("####################current step is %d/%d" % (current_step, steps))
						train(
							windows=5, model="A", epoch=150, batch_size=size, is_balance=True,
							num_filt_1=filt_1, num_core_1=core_1, num_fc_1=fc_1, is_show_result=False)
	else:
		train(windows=7, model="A", epoch=50, batch_size=64, is_show_input=False, is_Diff=False,
			is_balance=True, num_filt_1=7, num_core_1=6, num_fc_1=120, is_show_step_result=True)
