package lossfunction

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Calculates loss function for networks targets and outputs
func LossFunction(targets, outputs *mat.Dense) float64 {
	r, c := targets.Dims()
	loss := mat.NewDense(r, c, nil)
	// loss = (targets - outputs)
	loss.Sub(targets, outputs)
	// loss **2
	loss.Apply(func(r, c int, v float64) float64 { return math.Pow(v, 2.0) }, loss)
	// average all the elements in loss
	sum := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += loss.At(i, j)
		}
	}
	return sum / float64(r*c)
}

// Calculates loss function derivative
func LossFunctionDerivatives(targets, outputs *mat.Dense) *mat.Dense {
	r, c := targets.Dims()
	derivative := mat.NewDense(r, c, nil)
	derivative.Sub(targets, outputs)
	// -2 * (target - output)
	derivative.Apply(func(r, c int, v float64) float64 { return -2.0 * v }, derivative)
	return derivative
}
