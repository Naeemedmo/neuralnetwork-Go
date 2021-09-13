package main

import (
	"errors"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// shuffles the rows of any number of mat.Dense as long as they all have same number of rows
func ShuffleMatrices(input ...mat.Dense) {
	// check if all have the same number of rows
	row, _ := input[0].Dims()
	num_inputs := 0
	for i := range input {
		row_new, _ := input[i].Dims()
		if row_new != row {
			errors.New("Shuffle failed! Inputs have different number of rows!")
		}
		num_inputs++
	}
	// copy all the inputs
	tmp_input := make([]mat.Dense, num_inputs)
	for i := 0; i < num_inputs; i++ {
		tmp_input[i] = *mat.DenseCopyOf(&input[i])
		//input[i].Reset()
	}

	// Fisher–Yates shuffle
	hat := make([]bool, row)
	for i := range hat {
		hat[i] = false
	}
	IsHatEmpty := func(hat []bool) bool {
		for _, value := range hat {
			if !value {
				return false
			}
		}
		return true
	}
	new_row_number := 0
	for !IsHatEmpty(hat) {
		random_number := rand.Intn(row)
		if !hat[random_number] {
			for i := 0; i < num_inputs; i++ {
				// get the value
				data_row := tmp_input[i].RawRowView(random_number)
				input[i].SetRow(new_row_number, data_row)
			}
			hat[random_number] = true
			new_row_number++
		}
	}
}

// returns a 2d matrix with random values
func RadomMatrix(row, column int) mat.Dense {
	data := make([]float64, row*column)
	for i := range data {
		data[i] = rand.Float64()
	}
	return *mat.NewDense(row, column, data)
}
