package main

import (
	"fmt"
	"neuralnetwork/activationfunction"

	"gonum.org/v1/gonum/mat"
)

func main() {
	a := new([]int)
	*a = []int{4, 5}
	inputs4Network := NeuralNetwork{
		numInputs:          2,
		numOutputs:         2,
		hiddenLayers:       a,
		activationFunction: activationfunction.HyperbolicTangent,
	}
	row := 1000
	column := 2
	inputs := RadomMatrix(row, column)
	inputs.Apply(func(r, c int, v float64) float64 { return v / 2.0 }, &inputs)
	targets := mat.NewDense(row, column, nil)
	for i := 0; i < row; i++ {
		targets.Set(i, 0, inputs.At(i, 0)+inputs.At(i, 1))
		targets.Set(i, 1, inputs.At(i, 0)-inputs.At(i, 1))
	}
	// training
	neuralNetwork := InitializeNetwork(inputs4Network)
	epoch := 1000
	loss := make([]float64, epoch)
	for i := 0; i < epoch; i++ {
		loss_per_epoch := neuralNetwork.Train(inputs, *targets, 0.01, 10)
		loss = append(loss, loss_per_epoch)
		fmt.Println(i, loss_per_epoch)
	}

	// test
	fmt.Println("Start testing")
	num_test := 5
	test_inputs := RadomMatrix(num_test, 2)
	test_inputs.Apply(func(r, c int, v float64) float64 { return v / 2.0 }, &test_inputs)
	test_targets := mat.NewDense(num_test, column, nil)
	for i := 0; i < num_test-1; i++ {
		test_targets.Set(i, 0, test_inputs.At(i, 0)+test_inputs.At(i, 1))
		test_targets.Set(i, 1, test_inputs.At(i, 0)-test_inputs.At(i, 1))
	}
	test_outputs := neuralNetwork.Predict(test_inputs)
	for i := 0; i < num_test-1; i++ {
		fmt.Println("Test number: ", i)
		fmt.Println("Input: ", inputs.RawRowView(i))
		fmt.Println("Prediction: ", test_outputs.RawRowView(i))
		fmt.Println("Targets: ", test_targets.RawRowView(i))
	}
}
