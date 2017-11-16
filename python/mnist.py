import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import pickle
from scipy import misc

def process (x, thr_and_val):
	if x < 128:
		return -1
	return 1

thresholds_and_values = [[128, -1], [256, 1]]

input_filenames = ['mnist_train.csv', 'mnist_test.csv']
input_sizes=[60000, 10000]
output_filenames_img = ['img_train.dat', 'img_test.dat']
output_filenames_lbl = ['lbl_train.dat', 'lbl_test.dat']

for which in [0,1]:
	if which == 0:
		print('Processing training set.')
	else:
		print('Processing test set.')
		
	csvfile = open(input_filenames[which], 'r')
	reader = csv.reader(csvfile)
	new_size = 16
	images = np.zeros((input_sizes[which], new_size*new_size))
	labels = []
	count = 0

	start = time.time()

	for row in reader:
		tmpList = list(row)
		tmpList = [int(item) for item in tmpList]
		labels.append(tmpList[0])
		as_array = np.reshape(np.asarray(tmpList[1:]), (28,28))
		cropped = misc.imresize(as_array, (new_size, new_size), interp="bicubic")   # resize the image
		
		images[count] = np.reshape(cropped, (1, new_size*new_size))

		count += 1
		if count%100 == 0:
			print('Processed {} images...'.format(count))

		if count < 10*10+1:
			plt.subplot(10, 10, count)
			plt.title(labels[count-1])
			plt.xticks([])
			plt.yticks([])
			plt.imshow(np.reshape(images[count-1], (new_size, new_size)))
			plt.subplots_adjust(hspace=1, wspace=1)
		else:
			plt.show()
	csvfile.close()

	# Binarize
	for item in np.nditer(images, op_flags=['readwrite']):   # process the array
	    if item < 127:
	        item[...] = -1
	    else:
	        item[...] = 1	

	end = time.time()
	elapsed = end - start
	print('Elapsed time: {} seconds'.format(elapsed))


	# write processed data to file
	with open(output_filenames_img[which], 'wb') as outImg:
		pickle.dump(images, outImg)
	with open(output_filenames_lbl[which], 'wb') as outLbl:
		pickle.dump(labels, outLbl)
