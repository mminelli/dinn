## ----------------------------------------------------------------------------|
## Title      : Fast Homomorphic Evaluation of Deep Discretized Neural Networks
## Project    : Demonstrate Fast Fully Homomorphic Evaluation of Encrypted Inputs
##              using Deep Discretized Neural Networks hence preserving Privacy
## ----------------------------------------------------------------------------|
## File       : hom-mnist.py
## Authors    : Florian Bourse      <Florian.Bourse@ens.fr>
##              Michele Minelli     <Michele.Minelli@ens.fr>
##              Matthias Minihold   <Matthias.Minihold@RUB.de>
##              Pascal Paillier     <Pascal.Paillier@cryptoexperts.com>
//
## Reference  : TFHE: Fast Fully Homomorphic Encryption Library over the Torus
##              https://github.com/tfhe
## ----------------------------------------------------------------------------|
## Description:
##     Showcases how to efficiently evaluate privacy-perserving neural networks.
## ----------------------------------------------------------------------------|
## Revisions  :
## Date        Version  Description
## 2017-11-16  0.3.0    Version for github, referenced by ePrint paper
## ----------------------------------------------------------------------------|

import numpy as np
import pickle

# Load input file with images
with open('img_test.dat', 'rb') as in_img:
    images = pickle.load(in_img)

 # Preparation of the application dependent neural network
with open('lbl_test.dat', 'rb') as in_lbl:
    labels = pickle.load(in_lbl)
with open('start_biases.dat', 'rb') as in_biases:
    biases = pickle.load(in_biases)
with open('start_weights.dat', 'rb') as in_weights:
    weights = pickle.load(in_weights)

bias_scaling = 10
weight_scaling = 10
biases_1toN = np.round(biases[1][0]* bias_scaling)
biases_last = np.round(biases[2][0]* bias_scaling)

weights_1toN = np.round(weights[1] * weight_scaling)
weights_last = np.round(weights[2] * weight_scaling)

with open('txt_img_test.txt', 'wt') as out_img:
    for img in images:
        for pixel in img:
            out_img.write(str(np.int(pixel))+'\n')

with open('txt_labels.txt', 'wt') as out_lbl:
	for label in labels:
		out_lbl.write(str(np.int(label))+'\n')

with open('txt_weights.txt', 'wt') as out_weights:
	for i in range(weights[1].shape[0]):
		for j in range(weights[1].shape[1]):
			out_weights.write(str(np.int(np.round(weights[1][i][j] * weight_scaling)))+'\n')
	for i in range(weights[2].shape[0]):
		for j in range(weights[2].shape[1]):
			out_weights.write(str(np.int(np.round(weights[2][i][j] * weight_scaling)))+'\n')


with open('txt_biases.txt', 'wt') as out_biases:
	for i in range(biases[1].shape[1]):
		out_biases.write(str(np.int(np.round(biases[1][0][i] * bias_scaling)))+'\n')
	for i in range(biases[2].shape[1]):
		out_biases.write(str(np.int(np.round(biases[2][0][i] * bias_scaling)))+'\n')
