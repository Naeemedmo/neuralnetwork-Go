package neuralnetwork

import (
	"gonum.org/v1/gonum/mat"
)

// Calculates the activation for each neurons in the whole network
func (n *NeuralNetwork) FeedForward(inputs *mat.Dense) *[]mat.Dense {
	// There is an activation for each neurons of each layer
	activations := new([]mat.Dense)
	*activations = append(*activations, *inputs)
	// go through all the layers
	for l := 0; l < n.numLayers-1; l++ {
		addBias := func(r, c int, v float64) float64 {
			return v + n.biases[l].At(c, 0)
		}
		actFunc := func(r, c int, v float64) float64 {
			return n.ActivationFunction.Evaluate(v)
		}
		act := new(mat.Dense)
		act.Mul(&(*activations)[l], &n.weights[l])
		act.Apply(addBias, act)
		act.Apply(actFunc, act)
		*activations = append(*activations, *act)
	}

	return activations

}

// Predicts the output (:math:`a_{i}`) for a given input (:math:`a_{0}`)
func (n *NeuralNetwork) Predict(inputs *mat.Dense) *mat.Dense {
	activations := n.FeedForward(inputs)
	return &(*activations)[n.numLayers-1]
}
