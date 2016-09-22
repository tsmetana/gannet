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
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"time"
)

// The neural network
type NNet struct {
	Layers [][]*Neuron
}

// Createes a new network. The argument is an array of integers specifying the number of neurons in each layer. First
// array member is considered to be the input layer, the last number output layer. Second argument is the activation function
func NewNNet(layers []int, fun ActivationFunc) *NNet {
	var net NNet
	var i, j, k int
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Prepare the layers slices
	net.Layers = make([][]*Neuron, len(layers))
	// Allocate neuron pointers in each layer
	for i = 0; i < len(layers); i++ {
		net.Layers[i] = make([]*Neuron, layers[i])
	}
	// Initialize the neurons
	for i = 0; i < len(layers); i++ {
		for j = 0; j < len(net.Layers[i]); j++ {
			neuron := NewNeuron(1.0, fun)
			net.Layers[i][j] = neuron
			if i > 0 {
				// not the input layer: connect the neurons of the previous layer
				neuron.Weights = make([]float64, len(net.Layers[i-1]))
				neuron.Inputs = make([]*Neuron, len(net.Layers[i-1]))
				neuron.DeltaW = make([]float64, len(net.Layers[i-1]))
				for k = 0; k < len(net.Layers[i-1]); k++ {
					// connect with each neuron of the previous layer
					neuron.Inputs[k] = net.Layers[i-1][k]
					// initialize weights to a random number in (0, 0.1)
					neuron.Weights[k] = r.Float64()
				}
			} else {
				// input layer -- just one weight/input synapse per neuron and nil Inputs
				neuron.Weights = make([]float64, 1)
				neuron.DeltaW = make([]float64, 1)
				neuron.Weights[0] = 1.0
				// initialize weights to 1.0
				//neuron.Weights[k] = 1.0
			}
		}
	}

	return &net
}

// Training callback: called after each training iteration with the current error value passed in the argument.
// The return value of false indicates the training should stop.
type TrainingCBFunc func(err float64) bool

// Helper function
func (net *NNet) zeroErrors() {
	for i := 0; i < len(net.Layers); i++ {
		for j := 0; j < len(net.Layers[i]); j++ {
			net.Layers[i][j].Error = 0.0
		}
	}
}

// Computes output error and output layer gradients and weights for the given part of the output layer
// Returns the sum of the output errors through the channel: this also denotes end of the computation
func (net *NNet) computeOutputLayer(label []float64) float64 {
	var err_sum float64 = 0.0
	var neuron *Neuron

	layer_num := len(net.Layers) - 1
	for j := 0; j < len(net.Layers[layer_num]); j++ {
		neuron = net.Layers[layer_num][j]
		err_sum += (label[j] - neuron.Out) * (label[j] - neuron.Out) / 2
		neuron.Error = neuron.Activation.Prime(neuron.In) * (label[j] - neuron.Out)
		// compute lower layer errors
		neuron.update()
	}

	return err_sum
}

// Propagates error to the hidden//input layers, recomputes weights for the given part of the layer
// Writes into the 'done' channel when finished
func (net *NNet) propagateErrors(layer_num int) {
	var neuron *Neuron
	for j := 0; j < len(net.Layer[layer_num]); j++ {
		neuron = net.Layers[layer_num][j]
		neuron.update()
	}
}

// Computes one "epoch" of the training cycle: goest through the whole dataset in sequence
func (net *NNet) trainIteration(dataset []Dataset) (float64, error) {
	var i, j, d int

	output_error := 0.0
	// iterate over the dataset
	for d = 0; d < len(dataset); d++ {
		// run the d-th training example through the network
		net.feedForward(dataset[d].Data)
		net.zeroErrors()
		// traverse the net, compute errors, update weights
		for i = len(net.Layers) - 1; i >= 0; i-- {
			if i == len(net.Layers)-1 {
				net.computeOutputLayer(dataset[d].Label)
				output_error += net.computeOutputLayer(dataset[d].Label)
			} else {
				net.propagateErrors(i)
			}
		}
	}
	return output_error / float64(len(dataset)), nil
}

// Train the network on the given data set. Stops when the callback function returns false.
func (net *NNet) Train(dataset []Dataset, cb TrainingCBFunc) error {
	var keep_going bool = true
	var total_error float64
	var err error

	for keep_going {
		total_error, err = net.trainIteration(dataset)
		if err != nil {
			return err
		}
		// call the training callback function to see whether to stop
		keep_going = cb(total_error)
	}
	return nil
}

// The forward phase
func (net *NNet) feedForward(input []float64) {
	var i, j, k int
	var neuron *Neuron

	// Compute all the neuron outputs
	for i = 0; i < len(net.Layers); i++ {
		for j = 0; j < len(net.Layers[i]); j++ {
			neuron = net.Layers[i][j]
			neuron.In = -neuron.Bias
			if i > 0 {
				// hidden/output layer
				for k = 0; k < len(neuron.Weights); k++ {
					neuron.In += neuron.Weights[k] * neuron.Inputs[k].Out
				}
			} else {
				// input layer; use the input vector
				neuron.In += neuron.Weights[0] * input[j]
			}
			// apply the activation function
			neuron.Out = neuron.Activation.Activate(neuron.In)
		}
	}

}

// For given input vector computes the network's output
func (net *NNet) ComputeOutput(input []float64) ([]float64, error) {
	var i int

	// Check input usability
	if len(net.Layers[0]) != len(input) {
		return nil, errors.New(fmt.Sprintf("Error: input layer size: %d; input vector size: %d", len(net.Layers[0]), len(input)))
	}

	// Run the input vector through the net
	net.feedForward(input)
	// Copy the output layer to the return array
	ret := make([]float64, len(net.Layers[len(net.Layers)-1]))
	for i = 0; i < len(net.Layers[len(net.Layers)-1]); i++ {
		ret[i] = net.Layers[len(net.Layers)-1][i].Out
	}

	return ret, nil
}

// Sets the activation function for all the neurons in the network
// This is an workaround for the inability to save the activation (yet)
func (net *NNet) SetActivation(fun ActivationFunc) {
	var i, j int

	for i = 0; i < len(net.Layers); i++ {
		for j = 0; j < len(net.Layers[i]); j++ {
			net.Layers[i][j].Activation = fun
		}
	}
}

// Takes the JSON file and fills the NNet structure
func NNetLoad(filename string) (*NNet, error) {
	var json_net []byte
	var fd *os.File
	var err error = nil
	var new_net NNet
	var i, j, k int

	fd, err = os.Open(filename)
	defer fd.Close()
	if err != nil {
		return nil, err
	}
	st, err := fd.Stat()
	if err != nil {
		return nil, err
	}
	json_net = make([]byte, st.Size())
	_, err = fd.Read(json_net)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(json_net, &new_net)
	if err == nil {
		for i = 1; i < len(new_net.Layers); i++ {
			for j = 0; j < len(new_net.Layers[i]); j++ {
				new_net.Layers[i][j].Inputs = make([]*Neuron, len(new_net.Layers[i-1]))
				for k = 0; k < len(new_net.Layers[i-1]); k++ {
					new_net.Layers[i][j].Inputs[k] = new_net.Layers[i-1][k]
				}
			}
		}
	}
	new_net.setSegments()

	return &new_net, err
}

// Saves the NNet topology and weights to a JSON file.
func (net *NNet) Save(filename string) error {
	var json_net []byte
	var fd *os.File
	var err error = nil

	json_net, err = json.Marshal(net)
	if err != nil {
		return err
	}
	fd, err = os.Create(filename)
	defer fd.Close()
	if err != nil {
		return err
	}
	_, err = fd.Write(json_net)
	if err != nil {
		return err
	}

	return nil
}
