package neuralnetwork

import (
	"neuralnetwork-Go/extras"

	"gonum.org/v1/gonum/mat"
)

type NetworkProperties struct {
	NumInputs          int                ` Number of input neurons `
	HiddenLayers       *[]int             ` List of number of neurons in hidden layers`
	NumOutputs         int                ` Number of output neurons`
	ActivationFunction NetworkActivations ` Neuralnetwork's activation function`
}

type NeuralNetwork struct {
	NetworkProperties
	numLayers          int         ` Number of layers in the network`
	layers             []int       ` An array that represent the neurons in layers`
	weights            []mat.Dense ` Weight matrix between neurons of each two layers`
	biases             []mat.Dense ` Each neuron has a bias except input`
	weightsDerivatives []mat.Dense ` Derivatives of weights for training descent`
	biasesDerivatives  []mat.Dense ` Derivatives of biases for training the network`
}

// Initialize a neural network by setting all properties(layers, weights, biases
// and their derivatives)
func New(n NetworkProperties) NeuralNetwork {

	var nn NeuralNetwork

	nn.NetworkProperties = n
	nn.numLayers = len(*n.HiddenLayers) + 2
	nn.layers = append([]int{n.NumInputs}, *n.HiddenLayers...)
	nn.layers = append(nn.layers, n.NumOutputs)
	// weights and their derivatives
	for i := 0; i < nn.numLayers-1; i++ {
		matrix := extras.RadomMatrix(nn.layers[i], nn.layers[i+1])
		nn.weights = append(nn.weights, matrix)
	}
	for i := 0; i < nn.numLayers-1; i++ {
		matrix := extras.RadomMatrix(nn.layers[i], nn.layers[i+1])
		nn.weightsDerivatives = append(nn.weightsDerivatives, matrix)
	}

	// biases and their derivatives
	for i := 0; i < nn.numLayers-1; i++ {
		matrix := extras.RadomMatrix(nn.layers[i+1], 1)
		nn.biases = append(nn.biases, matrix)
	}
	for i := 0; i < nn.numLayers-1; i++ {
		matrix := extras.RadomMatrix(nn.layers[i+1], 1)
		nn.biasesDerivatives = append(nn.biasesDerivatives, matrix)
	}

	return nn
}
