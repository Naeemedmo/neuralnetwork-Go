package main

import (
	"neuralnetwork/lossfunction"

	"gonum.org/v1/gonum/mat"
)

// Calculates the new weights and biases derivatives for the network
func (n *InitializedNetwork) BackPropagate(targets mat.Dense, activations []mat.Dense) {

	// error = dC/da
	error := lossfunction.LossFunctionDerivatives(&targets, &activations[n.numLayers-1])
	// Walking backward to calculate derivatives
	for l := n.numLayers - 2; l >= 0; l-- {

		// delta = dC/da * da/dz
		delta := new(mat.Dense)
		delta = &activations[l+1]
		aFDerivative := func(r, c int, v float64) float64 {
			return n.activationFunction.Derivative(v)
		}
		delta.Apply(aFDerivative, delta)
		delta.MulElem(error, delta)

		// dC/db = dC/da * da/dz * dx/db [note that dx/db = 1]
		row, column := delta.Dims()

		for j := 0; j < column; j++ {
			average := 0.0
			for i := 0; i < row; i++ {
				average += delta.At(i, j)
			}
			average = average / float64(row)
			n.biasesDerivatives[l].Set(j, 0, average)
		}
		// dC/dw = dC/da * da/dz * dz/dw [note that dz/dw = a]
		n.weightsDerivatives[l].Mul(activations[l].T(), delta)
		// update error for next step
		// dC/da-1 = dC/da * da/dz * dz/da-1 [note that dz/da-1 = w]
		error.Reset()
		error.Mul(delta, n.weights[l].T())
	}

}
