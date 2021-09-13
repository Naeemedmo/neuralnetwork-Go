package main

import (
	"neuralnetwork/activationfunction"

	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	numInputs          int
	hiddenLayers       *[]int
	numOutputs         int
	activationFunction activationfunction.NetworkActivations
}

type InitializedNetwork struct {
	NeuralNetwork      ``
	numLayers          int         `Number of layers in the network`
	layers             []int       `An array that represent the neurons in layers`
	weights            []mat.Dense `Weight matrix between neurons of each two layers`
	biases             []mat.Dense `Each neuron has a bias except input`
	weightsDerivatives []mat.Dense `Derivatives of weights for training descent`
	biasesDerivatives  []mat.Dense `Derivatives of biases for training the network`
}

// Initialize a neural network by setting all properties(layers, weights, biases
// and their derivatives)
func InitializeNetwork(n NeuralNetwork) InitializedNetwork {

	var inn InitializedNetwork //internal neuralnetwork

	inn.NeuralNetwork = n
	inn.numLayers = len(*n.hiddenLayers) + 2
	inn.layers = append([]int{n.numInputs}, *n.hiddenLayers...)
	inn.layers = append(inn.layers, n.numOutputs)
	// weights and their derivatives
	for i := 0; i < inn.numLayers-1; i++ {
		matrix := RadomMatrix(inn.layers[i], inn.layers[i+1])
		inn.weights = append(inn.weights, matrix)
	}
	for i := 0; i < inn.numLayers-1; i++ {
		matrix := RadomMatrix(inn.layers[i], inn.layers[i+1])
		inn.weightsDerivatives = append(inn.weightsDerivatives, matrix)
	}

	// biases and their derivatives
	for i := 0; i < inn.numLayers-1; i++ {
		matrix := RadomMatrix(inn.layers[i+1], 1)
		inn.biases = append(inn.biases, matrix)
	}
	for i := 0; i < inn.numLayers-1; i++ {
		matrix := RadomMatrix(inn.layers[i+1], 1)
		inn.biasesDerivatives = append(inn.biasesDerivatives, matrix)
	}

	return inn
}
