package neuralnetwork

import (
	"neuralnetwork-Go/extras"
	"neuralnetwork-Go/lossfunction"

	"gonum.org/v1/gonum/mat"
)

// Trains the neural network
func (n *NeuralNetwork) Train(inputs, targets mat.Dense, learning_rate float64,
	batch_size int) float64 {
	sum_errors := 0.0
	// first shuffle input/target
	extras.ShuffleMatrices(inputs, targets)
	row, column := inputs.Dims()

	// Calculate number of batches
	n_batch := row / batch_size
	last_batch := batch_size
	if n_batch*batch_size != row { // last batch has less elements
		last_batch = row - n_batch*batch_size
	}
	for b := 0; b < n_batch; b++ {

		j := b * batch_size
		if b == n_batch-1 {
			batch_size = last_batch
		}

		sliced_input := mat.DenseCopyOf(inputs.Slice(j, j+batch_size, 0, column))
		sliced_targets := mat.DenseCopyOf(targets.Slice(j, j+batch_size, 0, column))
		// Calculate values for each neuron
		activations := n.FeedForward(*sliced_input)
		// Calculate the derivatives of weights/biases
		n.BackPropagate(*sliced_targets, activations)

		// Update weights/biases
		n.GradientDescent(learning_rate)
		// Calculate error
		sum_errors += lossfunction.LossFunction(sliced_targets, &activations[n.numLayers-1])

	}
	return sum_errors / float64(row)
}
