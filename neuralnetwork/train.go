package neuralnetwork

import (
	"neuralnetwork-Go/extras"
	"neuralnetwork-Go/lossfunction"

	"gonum.org/v1/gonum/mat"
)

// Trains the neural network
func (n *NeuralNetwork) Train(inputs, targets mat.Dense, learningRate float64,
	batchSize int) float64 {
	sumErrors := 0.0
	// first shuffle input/target
	extras.ShuffleMatrices(inputs, targets)
	row, column := inputs.Dims()

	// Calculate number of batches
	nBatch := row / batchSize
	LastBatch := batchSize
	if nBatch*batchSize != row { // last batch has less elements
		LastBatch = row - nBatch*batchSize
	}
	for b := 0; b < nBatch; b++ {

		j := b * batchSize
		if b == nBatch-1 {
			batchSize = LastBatch
		}

		slicedInputs := mat.DenseCopyOf(inputs.Slice(j, j+batchSize, 0, column))
		slicedTarget := mat.DenseCopyOf(targets.Slice(j, j+batchSize, 0, column))
		// Calculate values for each neuron
		activations := n.FeedForward(slicedInputs)
		// Calculate the derivatives of weights/biases
		n.BackPropagate(slicedTarget, activations)

		// Update weights/biases
		n.GradientDescent(learningRate)
		// Calculate error
		sumErrors += lossfunction.LossFunction(slicedTarget, &(*activations)[n.numLayers-1])

	}
	return sumErrors / float64(row)
}
