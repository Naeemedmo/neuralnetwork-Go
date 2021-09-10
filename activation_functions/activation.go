package activations

import (
	"errors"
	"math"
)

// Sigmoid function
// It takes any real value as input and outputs values in the range 0 to 1.
type Sigmoid struct{}

// :math:`S(x) = \\frac{1}{1+e^{-x}}`
func (s Sigmoid) evaluate() func(float64) float64 {
	return func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

}

// :math:`S'(x) = S(x)(1-S(x))`
func (s Sigmoid) derivative() func(float64) float64 {
	return func(x float64) float64 { return 1.0 / (1.0 + math.Exp(-s.input)) }
}

// Hyperbolic Tangent function.
// It takes any real value as input and outputs values in the range -1 to 1.
type HyperbolicTangent struct{}

// :math:`tanh(x) = \\frac{e^{x} - e^{-x}}{e^{x}+e^{-x}}`
func (h HyperbolicTangent) evaluate() func(float64) float64 {
	return func(x float64) float64 { return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x)) }
}

// :math:`tanh'(x) = 1-tanh(x)^2`
func (h HyperbolicTangent) derivative() func(float64) float64 {
	return func(x float64) float64 { return 1 - math.Pow(x, 2) }
}

// anything is a type activation function has evaluate/derivative
// An activation function has an evaluate and a derivative
type ActivationFunction interface {
	evaluate() func(float64) float64
	derivative() func(float64) float64
}

func Evaluate(a ActivationFunction) func(float64) float64 {
	activation, error := a.evaluate()
	if error != nil {
		errors.New("Function is not implemented!")
	} else {
		return activation
	}
}

func Derivative(a ActivationFunction) func(float64) float64 {
	derivative, error := a.derivative()
	if error != nil {
		errors.New("Function's derivative is not implemented!")
	} else {
		return derivative
	}
}
