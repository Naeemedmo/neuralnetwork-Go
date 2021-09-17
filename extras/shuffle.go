package extras

import (
	"errors"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Based on Fisher–Yates shuffle
// Shuffles the rows of any number of mat.Dense as long as they all have same
// number of rows
func ShuffleMatrices(input ...mat.Dense) {
	// check if all have the same number of rows
	row, _ := input[0].Dims()
	numInputs := 0
	for i := range input {
		nextRow, _ := input[i].Dims()
		if nextRow != row {
			errors.New("Shuffle failed! Inputs have different number of rows!")
		}
		numInputs++
	}
	// copy all the inputs
	tmpInput := make([]mat.Dense, numInputs)
	for i := 0; i < numInputs; i++ {
		tmpInput[i] = *mat.DenseCopyOf(&input[i])
		//input[i].Reset()
	}

	// Fisher–Yates shuffle
	hat := make([]bool, row)
	for i := range hat {
		hat[i] = false
	}
	isHatEmpty := func(hat []bool) bool {
		for _, value := range hat {
			if !value {
				return false
			}
		}
		return true
	}
	iRow := 0
	for !isHatEmpty(hat) {
		randomNumber := rand.Intn(row)
		if !hat[randomNumber] {
			for i := 0; i < numInputs; i++ {
				input[i].SetRow(iRow, tmpInput[i].RawRowView(randomNumber))
			}
			hat[randomNumber] = true
			iRow++
		}
	}
}
