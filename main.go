package main

import (
	"math/rand"
)

// array of random numbers
func RandomArray(length int) []float64 {
	var number float64
	random := make([]float64, 0)
	for l := 0; l < length; l++ {
		number = rand.Float64()
		random = append(random, number)
	}
	return random
}

type NeuralNetwork struct {
	numInputs          int
	hiddenLayers       [5]int
	numOutputs         int
	activationFunction string
}

// an array that represent the neurons in layers
func (n *NeuralNetwork) layers() []int {
	networkLayers := make([]int, 0)
	networkLayers = append(networkLayers, n.numInputs)
	for _, value := range n.hiddenLayers {
		if value > 0 {
			networkLayers = append(networkLayers, value)
		}
	}
	networkLayers = append(networkLayers, n.numOutputs)
	return networkLayers
}

// There is a weight matrix between each two layers with size of neurons on both side
func (n *NeuralNetwork) weights() [][][]float64 {
	networkWeights := make([][][]float64, len(n.layers())-1)
	for i := range networkWeights {
		networkWeights[i] = make([][]float64, n.layers()[i+1])
		for j := range networkWeights[i] {
			networkWeights[i][j] = append(networkWeights[i][j], RandomArray(n.layers()[i])...)
		}
	}
	return networkWeights
}

// Each activation has a bias except input
func (n *NeuralNetwork) biases() [][]float64 {
	networkBiases := make([][]float64, len(n.layers())-1)
	for i := range networkBiases {
		networkBiases[i] = append(networkBiases[i], RandomArray(n.layers()[i+1])...)
	}
	return networkBiases
}

func main() {
	neuraltest := NeuralNetwork{
		numInputs:          2,
		numOutputs:         2,
		hiddenLayers:       [...]int{4, 5, 0, 0, 0},
		activationFunction: "test",
	}

}
