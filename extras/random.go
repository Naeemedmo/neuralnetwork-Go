package extras

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// returns a 2d matrix with random values
func NewRadomMatrix(row, column int) mat.Dense {
	data := make([]float64, row*column)
	for i := range data {
		data[i] = rand.Float64()
	}
	return *mat.NewDense(row, column, data)
}
