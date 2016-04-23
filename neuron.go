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
	"reflect"
	"sync"
)

// The neuron
type Neuron struct {
	sync.Mutex                    // Lock for parallel comutation
	Bias           float64        // The neuron (initial bias
	In             float64        `json:"-"` // The exctitation; input of the activation function
	Out            float64        `json:"-"` // Output of the neuron
	Error          float64        `json:"-"` // Stores the value of error during backpropagation
	ActivationName string         // String description of the activation function
	Activation     ActivationFunc `json:"-"` // The actual activation function
	Weights        []float64      // Input synapses weights
	Inputs         []*Neuron      `json:"-"` // Pointers to connected neurons for traversals
	DeltaW         []float64      // The weight change
	LearningRate   float64        // Backprop parameter
	Momentum       float64        // Dtto
}

// The neuron "constructor"
func NewNeuron(b float64, fun ActivationFunc) *Neuron {
	return &Neuron{
		Bias:           b,
		Activation:     fun,
		ActivationName: reflect.TypeOf(fun).Name(),
		Momentum:       0.9,
	}
}

// Adjust the weights and propagate the error through all the "dendrits"
func (neuron *Neuron) update() {
	for k := 0; k < len(neuron.Weights); k++ {
		neuron.Inputs[k].Lock()
		neuron.Inputs[k].Error += neuron.Activation.Prime(neuron.Inputs[k].In) * neuron.Error * neuron.Weights[k]
		neuron.Inputs[k].Unlock()
		neuron.DeltaW[k] = neuron.Error*neuron.Inputs[k].Out + neuron.Momentum*neuron.DeltaW[k]
		neuron.Weights[k] += neuron.DeltaW[k]
	}
}
