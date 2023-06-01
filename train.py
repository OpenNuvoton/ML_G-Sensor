# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=redefined-outer-name
# pylint: disable=g-bad-import-order

"""Build and train neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
from data_load import DataLoader

import numpy as np
import tensorflow as tf

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
NUM_REP_DATA_SAMPLES = 100  # How many samples to use for post training quantization.


def reshape_function(data, label):
  reshaped_data = tf.reshape(data, [-1, 3, 1])
  return reshaped_data, label


def calculate_model_size(model):
  print(model.summary())
  var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
  ]
  print("Model size:", sum(var_sizes) / 1024, "KB")


def build_cnn(seq_length, out_dim):
  """Builds a convolutional neural network in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          8, (4, 3),
          padding="same",
          input_shape=(seq_length, 3, 1)),  # output_shape=(batch, 128, 3, 8)
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D((3, 3)),  # (batch, 42, 1, 8)
      tf.keras.layers.Dropout(0.1),  # (batch, 42, 1, 8)
      
      tf.keras.layers.Conv2D(16, (4, 1), padding="same"),  # (batch, 42, 1, 16)
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 14, 1, 16)
      
      tf.keras.layers.Conv2D(32, (4, 1), padding="same"),  # (batch, 42, 1, 16)
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
      tf.keras.layers.Dropout(0.5),  # (batch, 14, 1, 16)
      
      tf.keras.layers.GlobalAveragePooling2D(),
      #tf.keras.layers.Flatten(),  # (batch, 224)
      tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 16)
      tf.keras.layers.Dense(out_dim, activation="softmax")  # (batch, 4)
  ])
  
  model_path = os.path.join("./netmodels", "CNN")
  
  return model, model_path

def build_cnns(seq_length, out_dim):
  """Builds a convolutional neural network in Keras."""
  
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          8, (4, 3),
          padding="same",
          input_shape=(seq_length, 3, 1)),  # output_shape=(batch, 128, 3, 8)
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D((3, 3)),  # (batch, 42, 1, 8)
      tf.keras.layers.Dropout(0.1),  # (batch, 42, 1, 8)
      
      tf.keras.layers.Conv2D(16, (4, 1), padding="same"),  # (batch, 42, 1, 16)
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation("relu"),
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 14, 1, 16)
      
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 16)
      tf.keras.layers.Dense(out_dim, activation="softmax")  # (batch, 4)
  ])
  
  model_path = os.path.join("./netmodels", "CNN")
  
  return model, model_path


def build_lstm(seq_length, out_dim):
  """Builds an LSTM in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(22),
          input_shape=(seq_length, 3)),  # output_shape=(batch, 44)
      tf.keras.layers.Dense(out_dim, activation='sigmoid')  # (batch, 4)
  ])
  model_path = os.path.join("./netmodels", "LSTM")
  
  return model, model_path


def load_data(train_data_path, valid_data_path, test_data_path, seq_length, folders_name):
  data_loader = DataLoader(
      train_data_path, valid_data_path, test_data_path, folders_name, seq_length=seq_length)
  data_loader.format()
  return data_loader.train_len, data_loader.train_data, data_loader.valid_len, \
      data_loader.valid_data, data_loader.test_len, data_loader.test_data

def build_net(args, seq_length, out_dim):
  if args.model == "CNN":
    model, model_path = build_cnn(seq_length, out_dim)
  elif args.model == "CNN-S":
    model, model_path = build_cnns(seq_length, out_dim)  
  elif args.model == "LSTM":
    model, model_path = build_lstm(seq_length, out_dim)
  else:
    print("Please input correct model name.(CNN CNN-S LSTM)")
  return model, model_path



def convert(model_path, train_data, model_type):
    
    # representative dataset
    train_data = train_data.map(reshape_function)
    def _rep_dataset():
        """Generator function to produce representative dataset."""
        i = 0
        for data in train_data.batch(1).take(NUM_REP_DATA_SAMPLES): # [((1, 128, 3, 1), 1), ..., ((1, 128, 3, 1), 1)] => ... = take number  
            if i > NUM_REP_DATA_SAMPLES:
                break
            i += 1
            yield [tf.dtypes.cast(data[0], tf.float32)] # data[0] => (1, 128, 3, 1) real 1 input
  
    # load keras model
    model = tf.keras.models.load_model(model_path)
    
    # Convert the model to the TensorFlow Lite format without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # Save the model to disk
    open("./generated_model/model_{}.tflite".format(model_type), "wb").write(tflite_model)
  
    # Convert the model to the TensorFlow Lite format with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _rep_dataset
    tflite_model = converter.convert()
    # Save the model to disk
    open("./generated_model/model_{}_quantized.tflite".format(model_type), "wb").write(tflite_model)
  
    basic_model_size = os.path.getsize("./generated_model/model_{}.tflite".format(model_type))
    print("Basic model is %d bytes" % basic_model_size)
    quantized_model_size = os.path.getsize("./generated_model/model_{}_quantized.tflite".format(model_type))
    print("Quantized model is %d bytes" % quantized_model_size)
    difference = basic_model_size - quantized_model_size
    print("Difference is %d bytes" % difference)  
  

def train_net(
    model,
    model_path,  # pylint: disable=unused-argument
    train_len,  # pylint: disable=unused-argument
    train_data,
    valid_len,
    valid_data,  # pylint: disable=unused-argument
    test_len,
    test_data,
    kind,
    epochs,
    batch_size,
    out_dim):
  """Trains the model."""
  calculate_model_size(model)
  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"])
      
  if kind == "CNN" or kind == "CNN-S":
    train_data = train_data.map(reshape_function)
    test_data = test_data.map(reshape_function)
    valid_data = valid_data.map(reshape_function)
    
  test_labels = np.zeros(test_len)
  idx = 0
  for data, label in test_data:  # pylint: disable=unused-variable
    test_labels[idx] = label.numpy()
    idx += 1
  train_data = train_data.batch(batch_size).repeat()
  valid_data = valid_data.batch(batch_size)
  test_data = test_data.batch(batch_size)
  
  model.fit(
      train_data,
      epochs=epochs,
      validation_data=valid_data,
      steps_per_epoch=1000,
      validation_steps=int((valid_len - 1) / batch_size + 1),
      )
      
  loss, acc = model.evaluate(test_data)
  pred = np.argmax(model.predict(test_data), axis=1)
  confusion = tf.math.confusion_matrix(
      labels=tf.constant(test_labels),
      predictions=tf.constant(pred),
      num_classes = out_dim)
  print(confusion)
  print("Loss {}, Accuracy {}".format(loss, acc))
  
  # Save the model as h5
  model_path = os.path.join("./netmodels", kind)
  print("Built {}.".format(kind))
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  model.save("./netmodels/{}/weights.h5".format(kind))

def test_tflite(tflite_path, test_data, test_len, out_dim):
  """Test the tflite model."""
  
  test_data = test_data.map(reshape_function)

  #expected_indices = np.concatenate([y for x, y in test_data])
  predicted_indices = []
  
  test_labels = np.zeros(test_len)
  idx = 0
  for data, label in test_data:  # pylint: disable=unused-variable
    test_labels[idx] = label.numpy()
    idx += 1
    prediction = tflite_inference(tf.expand_dims(data, axis=0), tflite_path)
    predicted_indices.append(np.squeeze(tf.argmax(prediction, axis=1)))

  test_accuracy = calculate_accuracy(predicted_indices, test_labels)
  confusion_matrix = tf.math.confusion_matrix(labels=tf.constant(test_labels), 
                                              predictions=predicted_indices,
                                              num_classes = out_dim)
  print(confusion_matrix.numpy())
  print(f'Test accuracy = {test_accuracy * 100:.2f}%')
   
def tflite_inference(input_data, tflite_path):
    """Call forwards pass of TFLite file and returns the result.

    Args:
        input_data: Input data to use on forward pass.
        tflite_path: Path to TFLite file to run.

    Returns:
        Output from inference.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    # Check if the input/output type is quantized,
    # set scale and zero-point accordingly
    if input_dtype == np.int8:
        input_scale, input_zero_point = input_details[0]["quantization"]
    else:
        input_scale, input_zero_point = 1, 0

    input_data = input_data / input_scale + input_zero_point
    input_data = np.round(input_data) if input_dtype == np.int8 else input_data

    if output_dtype == np.int8:
        output_scale, output_zero_point = output_details[0]["quantization"]
    else:
        output_scale, output_zero_point = 1, 0

    interpreter.set_tensor(input_details[0]['index'], tf.cast(input_data, input_dtype))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

    return output_data

def calculate_accuracy(predicted_indices, expected_indices):
    """Calculates and returns accuracy.

    Args:
        predicted_indices: List of predicted integer indices.
        expected_indices: List of expected integer indices.

    Returns:
        Accuracy value between 0 and 1.
    """
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", help='The model type, CNN, CNN-S')
  parser.add_argument(
        '--folders',
        type=str,
        nargs="+",
        default=["ring", "slope", "wing"],
        help='Read data from folders, ex: "/slope", "/ring"')
  parser.add_argument(
        '--out_dir',
        type=str,
        default='out_dataset_1',
        help='What dataset to be used')
  parser.add_argument(
        '--seq_length',
        type=int,
        default=128,
        help='Decide the feature number of 1 dim')
  parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='The training epochs')
  parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='The batch size')
  parser.add_argument(
        '--convert_only',
        type=int,
        default=0,
        help='1: Only convert h5 to tflite')
  parser.add_argument(
        '--test_tflite',
        type=bool,
        default=False,
        help='Test the tflite file only')
  parser.add_argument(
        '--tflite_path',
        type=str,
        default='model_CNN-S_quantized.tflite',
        help='The tflite for testing')            
  args = parser.parse_args()

  print("Start to load data...")
 
  train_len, train_data, valid_len, valid_data, test_len, test_data = \
  load_data("./data/%s/train" % (args.out_dir), "./data/%s/valid" % (args.out_dir), "./data/%s/test" % (args.out_dir), args.seq_length, args.folders)
  
  keras_model_pth = "./netmodels/{}/weights.h5".format(args.model)
  out_dim = len(args.folders) + 1

  if args.test_tflite:
      print("Start testing...")
      test_tflite(args.tflite_path, test_data, test_len, out_dim)
  else:
      if args.convert_only:
          convert(keras_model_pth, train_data, args.model)
      else:
          print("Start to build net...")
          model, model_path = build_net(args, args.seq_length, out_dim)
        
          print("Start training...")
          print("Actual data number Train:{} Val:{} Test:{}".format(train_len, valid_len, test_len))
          train_net(model, model_path, train_len, train_data, valid_len, valid_data,
                    test_len, test_data, args.model, args.epochs, args.batch_size, out_dim)
          convert(keras_model_pth, train_data, args.model)
          print("Training finished!")
