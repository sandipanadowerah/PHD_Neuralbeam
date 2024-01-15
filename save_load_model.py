#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to save and load a model structure and weights.
Code taken from
https://machinelearningmastery.com/save-load-keras-deep-learning-models/
"""
import keras.models


def save_model(model, name, verbose=0):
    """
    Save the model structure into json file and the weights to HDF5
    :param model:       model to store
    :param name:        name of the files (path included if not in working directory)
    :param verbose:     verbose [0]
    :return:            Two files are saved under name.json and name.h5
    """
    model_json = model.to_json()
    with open(name + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + '.h5')
    if verbose:
        print("Saved model to disk")


def load_model(name, verbose=0):
    """
    Load a model structure and weights from the files 'name.json' (structure) and 'name.h5' (weights)
    :param name:        name of the file without extension (should be the same name for structure and weights;
                                                            path included if not in working directory)
    :param verbose:     verbose [0]
    :return: model:     model loaded
    """
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + '.h5')
    if verbose:
        print('Loaded model from disk')
    return loaded_model
