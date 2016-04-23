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
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"testing"
)

type outputTestData struct {
	inVal  float64
	outErr float64
}

// globals to be shared among the tests
var net *NNet
var ds []Dataset
var epoch int
var skip_output_test bool
var out_data []outputTestData
var net_file string

const ds_size int = 1000
const epoch_max int = 200
const output_cases int = 20

func TestMain(m *testing.M) {

	ds = make([]Dataset, ds_size)

	// Prepare the dataset: sin function in (0, pi), no need to scale output
	for i := 0; i < ds_size; i++ {
		ds[i].Data = make([]float64, 1)
		ds[i].Data[0] = rand.Float64()
		ds[i].Label = make([]float64, 1)
		ds[i].Label[0] = math.Sin(ds[i].Data[0] * math.Pi)
	}

	// Prepare the testing data: need to store the errors made during computation
	// hence not using the Dataset type
	out_data = make([]outputTestData, output_cases)
	for i := 0; i < output_cases; i++ {
		out_data[i].inVal = rand.Float64()
	}
	ret := m.Run()

	os.Remove(net_file)
	os.Exit(ret)
}

func trainingCb(net_err float64) bool {
	// Stop after certain number of epochs
	epoch++

	if epoch == epoch_max {
		return false
	}
	return true
}

func TestNewNNet_181(t *testing.T) {
	var s Sigmoid
	net = NewNNet([]int{1, 8, 1}, &s)
}

func TestTrain_181(t *testing.T) {
	err := net.Train(ds, trainingCb)
	epoch = 0
	if err != nil {
		t.Fatal(err)
	}
}

func TestComputeOutput_181(t *testing.T) {
	epoch = 0
	for i := 0; i < output_cases; i++ {
		out, err := net.ComputeOutput([]float64{out_data[i].inVal})
		if err != nil {
			t.Errorf("Output computation failed for %f\n", out_data[i].inVal)
		} else {
			out_data[i].outErr = out[0] - math.Sin(out_data[i].inVal*math.Pi)
			t.Logf("Ouptut for %f: %f; Expected %f (Error %f)\n", out_data[i].inVal, out[0], math.Sin(out_data[i].inVal*math.Pi), out_data[i].outErr)
		}
	}
}

func TestSave(t *testing.T) {
	tmpfile, err := ioutil.TempFile("", "nntest")
	if err != nil {
		t.Error(err)
	}
	net_file = tmpfile.Name()
	tmpfile.Close()
	t.Log("Created temporary file:", net_file)

	err = net.Save(net_file)
	if err != nil {
		t.Error(err)
	}
}

func TestNNetLoad(t *testing.T) {
	var s Sigmoid

	net, err := NNetLoad(net_file)
	if err != nil {
		t.Error(err)
	}
	epoch = 0 // reset the epoch counter
	net.SetActivation(&s)
	for i := 0; i < output_cases; i++ {
		out, err := net.ComputeOutput([]float64{out_data[i].inVal})
		if err != nil {
			t.Fail()
		} else {
			new_err := out[0] - math.Sin(out_data[i].inVal*math.Pi)
			if out_data[i].outErr != new_err {
				t.Errorf("Ouptut for %f differs from previous run: %f != %f )\n", out_data[i].inVal, new_err, out_data[i].outErr)
			}
		}
	}
}
