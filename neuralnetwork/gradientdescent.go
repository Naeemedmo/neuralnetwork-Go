package neuralnetwork

//Updates the weights and biases with derivative of loss function
func (n *NeuralNetwork) GradientDescent(learning_rate float64) {

	// loop over all the layers
	for l := 0; l < n.numLayers-1; l++ {

		// update the weights
		row, column := n.weights[l].Dims()
		for i := 0; i < row; i++ {
			for j := 0; j < column; j++ {
				updated := n.weights[l].At(i, j) - n.weightsDerivatives[l].At(i, j)*learning_rate
				n.weights[l].Set(i, j, updated)
			}
		}
		// update biases
		row, column = n.biases[l].Dims()
		for i := 0; i < row; i++ {
			for j := 0; j < column; j++ {
				updated := n.biases[l].At(i, j) - n.biasesDerivatives[l].At(i, j)*learning_rate
				n.biases[l].Set(i, j, updated)
			}
		}

	}
}
