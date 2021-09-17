package neuralnetwork

import (
	"math"
)

type AFunction func(float64) float64

type NetworkActivations struct {
	Evaluate   AFunction
	Derivative AFunction
}

// Linear function
// Function: :math:`F(x) = x`
// Functions derivative: :math:`F'(x) = 1.0`
var Linear NetworkActivations = NetworkActivations{
	func(x float64) float64 { return x },
	func(x float64) float64 { return 1.0 },
}

// Sigmoid function
// It takes any real value as input and outputs values in the range 0 to 1.0
// Function: :math:`S(x) = \\frac{1}{1+e^{-x}}`
// Functions derivative: :math:`S'(x) = S(x)(1-S(x))`
var Sigmoid NetworkActivations = NetworkActivations{
	func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) },
	func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) },
}

// Hyperbolic tangent function
// It takes any real value as input and outputs values in the range -1 to 1.
// Function: :math:`tanh(x) = \\frac{e^{x} - e^{-x}}{e^{x}+e^{-x}}`
// Functions derivative: :math:`tanh'(x) = 1-tanh(x)^2`
var HyperbolicTangent NetworkActivations = NetworkActivations{
	func(x float64) float64 { return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x)) },
	func(x float64) float64 { return 1 - math.Pow(x, 2) },
}
