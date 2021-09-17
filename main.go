package main

import (
	"fmt"
	"neuralnetwork-Go/extras"
	"neuralnetwork-Go/neuralnetwork"

	"gonum.org/v1/gonum/mat"
)

func main() {
	networkProperties := neuralnetwork.NetworkProperties{
		NumInputs:          2,
		HiddenLayers:       &[]int{4, 5},
		NumOutputs:         2,
		ActivationFunction: neuralnetwork.HyperbolicTangent,
	}
	row := 1000
	column := 2
	inputs := extras.NewRadomMatrix(row, column)
	inputs.Apply(func(r, c int, v float64) float64 { return v / 2.0 }, &inputs)
	targets := mat.NewDense(row, column, nil)
	for i := 0; i < row; i++ {
		targets.Set(i, 0, inputs.At(i, 0)+inputs.At(i, 1))
		targets.Set(i, 1, inputs.At(i, 0)-inputs.At(i, 1))
	}
	// training
	neuralNetwork := neuralnetwork.New(networkProperties)
	epoch := 1000
	loss := make([]float64, epoch)
	for i := 0; i < epoch; i++ {
		lossPerEpoch := neuralNetwork.Train(inputs, *targets, 0.01, 10)
		loss = append(loss, lossPerEpoch)
		fmt.Println(i, lossPerEpoch)
	}

	// test
	fmt.Println("Start testing")
	numTest := 5
	testInputs := extras.NewRadomMatrix(numTest, 2)
	testInputs.Apply(func(r, c int, v float64) float64 { return v / 2.0 }, &testInputs)
	testTargets := mat.NewDense(numTest, column, nil)
	for i := 0; i < numTest-1; i++ {
		testTargets.Set(i, 0, testInputs.At(i, 0)+testInputs.At(i, 1))
		testTargets.Set(i, 1, testInputs.At(i, 0)-testInputs.At(i, 1))
	}
	testOutputs := neuralNetwork.Predict(&testInputs)
	for i := 0; i < numTest-1; i++ {
		fmt.Println("Test number:  ", i)
		fmt.Println("Input:        ", inputs.RawRowView(i))
		fmt.Println("Prediction:   ", testOutputs.RawRowView(i))
		fmt.Println("Targets:      ", testTargets.RawRowView(i))
	}
}
