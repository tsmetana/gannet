/*

	GANNet: The Go Artificial Neural Network library

	Copyright (C) 2016  Tomas Smetana <tomas@smetana.name>

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License along
	with this program; if not, write to the Free Software Foundation, Inc.,
	51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

*/
package gannet

import (
	"math"
)

// Activation function interface (needs to compute the activation output as well as a diffrential in a given point)
type ActivationFunc interface {
	Activate(x float64) float64 // This is perhaps not the best terminology...
	Prime(x float64) float64
}

// Sigmoid function
type Sigmoid struct {
}

// Sigmoid function
func (f *Sigmoid) Activate(x float64) float64 {
	return (1.0 / (1.0 + math.Exp(-x)))
}

// Sigmoid function derivative
func (f *Sigmoid) Prime(x float64) float64 {
	fx := f.Activate(x)
	return (fx * (1.0 - fx))
}
