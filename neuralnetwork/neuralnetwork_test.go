package neuralnetwork

import (
	"fmt"
	"math"
	"neuralnetwork-Go/lossfunction"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Checks the derivatives of the network by a simple numerical differentiation method
func TestNetworkPerformance(t *testing.T) {
	// inputs
	epsilon := 1e-10
	singleInput := mat.NewDense(1, 2, []float64{0.3, 0.1})
	singleTarget := mat.NewDense(1, 1, []float64{0.4})
	networkProperties := NetworkProperties{
		NumInputs:          2,
		NumOutputs:         1,
		HiddenLayers:       &[]int{3},
		ActivationFunction: HyperbolicTangent,
	}
	// initialize the network
	neuralNetwork := New(networkProperties)
	activations := neuralNetwork.FeedForward(singleInput)
	error1 := lossfunction.LossFunction(singleTarget, &(*activations)[neuralNetwork.numLayers-1])
	// calculate derivatives once
	neuralNetwork.BackPropagate(singleTarget, activations)

	for i := 0; i < neuralNetwork.numLayers-1; i++ {
		for j := 0; j < neuralNetwork.layers[i]; j++ {
			for k := 0; k < neuralNetwork.layers[i+1]; k++ {
				// change the weight by epsilon (x + h)
				neuralNetwork.weights[i].Set(j, k, neuralNetwork.weights[i].At(j, k)+epsilon)
				//f(x+h), dC/dw
				output2 := neuralNetwork.Predict(singleInput)
				error2 := lossfunction.LossFunction(singleTarget, output2)
				// slope = (f(x+h) - f(x)) / h
				numericalDerivative := (error2 - error1) / epsilon
				difference := math.Abs(numericalDerivative - neuralNetwork.weightsDerivatives[i].At(j, k))
				// TEST
				want := 1e-5
				if got := difference; got > want {
					t.Errorf("Derivative check failed! Check network implementation!\n")
					fmt.Printf("Layer %d, Leftside neuron: %d, rightside neuron: %d\n", i, j, k)
					fmt.Printf("The difference is:              %5f\n", difference)
					fmt.Printf("The numerical derivative is:    %5f\n", numericalDerivative)
					fmt.Printf("Networks derivative is:         %5f\n", neuralNetwork.weightsDerivatives[i].At(j, k))
				}

			}
		}
	}
}
