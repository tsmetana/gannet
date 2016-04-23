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
	"os"
)

// The dataset
type Dataset struct {
	Data  []float64
	Label []float64
}

// Loads a dataset from a json file
func DatasetLoad(filename string) ([]Dataset, error) {
	var json_ds []byte
	var fd *os.File
	var err error = nil
	var new_ds []Dataset

	fd, err = os.Open(filename)
	defer fd.Close()
	if err != nil {
		return nil, err
	}
	st, err := fd.Stat()
	if err != nil {
		return nil, err
	}
	json_ds = make([]byte, st.Size())
	_, err = fd.Read(json_ds)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(json_ds, &new_ds)

	return new_ds, err
}

// Saves a dataset to a json file
// FIXME: *Dataset is definitely not the correct type
func (ds *Dataset) Save(filename string) error {
	var json_ds []byte
	var fd *os.File
	var err error = nil

	json_ds, err = json.Marshal(ds)
	if err != nil {
		return err
	}
	fd, err = os.Create(filename)
	defer fd.Close()
	if err != nil {
		return err
	}
	_, err = fd.Write(json_ds)
	if err != nil {
		return err
	}

	return nil
}
