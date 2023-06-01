# Lint as: python3
# coding=utf-8
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

"""Raw data folders are all put inside the data/, 
   and the output prepared data are inside data/{your_naming_dir}
"""

"""Prepare data for further process.

Read data from "data/slope", "data/ring", "data/wing", "data/negative" and save them
in "/data/{your_naming_dir}/complete_data" in python dict format.

It will generate a new file with the following structure:
├── data
│   └── {your_naming_dir}
│        └── complete_data
"""

"""Mix and split data.

Mix different people's data together and randomly split them into train,
validation and test. These data would be saved separately under "/data/{your_naming_dir}".
It will generate new files with the following structure:

├── data
│   └── {your_naming_dir}
│        ├── complete_data
│        ├── test
│        ├── train
│        └── valid
"""

import csv
import json
import os
import random
import argparse
import math

LABEL_NAME = "gesture"
DATA_NAME = "accel_ms2_xyz"

#folders = ["action1", "action2", "action3"]
#names = ["joseph"]

#folders = ["ring", "slope", "wing"]
#names = ["hyw", "shiyun", "tangsy", "dengyl", "zhangxy", "pengxl", "liucx", "jiangyh", "xunkai"]


def prepare_original_data(folder, name, data, file_to_read, seq_length):  # pylint: disable=redefined-outer-name
  """Read collected data from files."""
  if folder != "negative":
    with open(file_to_read, "r",encoding="utf-8") as f:
      lines = csv.reader(f)
      data_new = {}
      data_new[LABEL_NAME] = folder
      data_new[DATA_NAME] = []
      data_new["name"] = name
      for idx, line in enumerate(lines):  # pylint: disable=unused-variable,redefined-outer-name
        if len(line) == 3 or len(line) == 4:
          if line[2] == "-" and data_new[DATA_NAME]:
            data.append(data_new)
            data_new = {}
            data_new[LABEL_NAME] = folder
            data_new[DATA_NAME] = []
            data_new["name"] = name
          elif line[2] != "-":
            data_new[DATA_NAME].append([float(i) for i in line[0:3]])
      data.append(data_new)
  else:
    with open(file_to_read, "r",encoding="utf-8") as f:
      lines = csv.reader(f)
      data_new = {}
      data_new[LABEL_NAME] = folder
      data_new[DATA_NAME] = []
      data_new["name"] = name
      for idx, line in enumerate(lines):
        if len(line) == 3 or len(line) == 4:
          if line[2] == "-" and data_new[DATA_NAME]:
            data.append(data_new)
            data_new = {}
            data_new[LABEL_NAME] = folder
            data_new[DATA_NAME] = []
            data_new["name"] = name
          elif line[2] != "-":
            data_new[DATA_NAME].append([float(i) for i in line[0:3]])
        #if len(line) == 3 and line[2] != "-":
        #  if len(data_new[DATA_NAME]) == seq_length:
        #    data.append(data_new)
        #    data_new = {}
        #    data_new[LABEL_NAME] = folder
        #    data_new[DATA_NAME] = []
        #    data_new["name"] = name
        #    continue
        #  else:
        #    data_new[DATA_NAME].append([float(i) for i in line[0:3]])
      data.append(data_new)


def generate_negative_data(data, seq_length, neg_data_num, train_ratio, val_ratio):  # pylint: disable=redefined-outer-name
  """Generate negative data labeled as 'negative6~8'."""
  # Big movement -> around straight line
  for i in range(neg_data_num): # 100
    if i > math.floor(neg_data_num * (train_ratio + val_ratio)): # i>100*(0.6+0.2)
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > math.floor(neg_data_num * train_ratio): # i>100*0.6
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    start_x = (random.random() - 0.5) * 2000 #2000
    start_y = (random.random() - 0.5) * 2000 #2000
    start_z = (random.random() - 0.5) * 2000 #2000
    x_increase = (random.random() - 0.5) * 10 #10
    y_increase = (random.random() - 0.5) * 10 #10
    z_increase = (random.random() - 0.5) * 10 #10
    for j in range(seq_length):
      dic[DATA_NAME].append([
          start_x + j * x_increase + (random.random() - 0.5) * 6,
          start_y + j * y_increase + (random.random() - 0.5) * 6,
          start_z + j * z_increase + (random.random() - 0.5) * 6
      ])
    data.append(dic)
  # Random
  for i in range(neg_data_num):
    if i > math.floor(neg_data_num * (train_ratio + val_ratio)): # i>100*(0.6+0.2)
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > math.floor(neg_data_num * train_ratio): # i>100*0.6
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    for j in range(seq_length):
      dic[DATA_NAME].append([(random.random() - 0.5) * 1000, #1000
                             (random.random() - 0.5) * 1000, #1000
                             (random.random() - 0.5) * 1000])
    data.append(dic)
  # Stay still
  for i in range(neg_data_num):
    if i > math.floor(neg_data_num * (train_ratio + val_ratio)): # i>100*(0.6+0.2)
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > math.floor(neg_data_num * train_ratio): # i>100*0.6
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    start_x = (random.random() - 0.5) * 2000 #2000
    start_y = (random.random() - 0.5) * 2000 #2000
    start_z = (random.random() - 0.5) * 2000 #2000
    for j in range(seq_length):
      dic[DATA_NAME].append([
          start_x + (random.random() - 0.5) * 40, #40
          start_y + (random.random() - 0.5) * 40, #40
          start_z + (random.random() - 0.5) * 40 #40
      ])
    data.append(dic)


# Write data to file
def write_data(data_to_write, path):
  with open(path, "w") as f:
    for idx, item in enumerate(data_to_write):  # pylint: disable=unused-variable,redefined-outer-name
      dic = json.dumps(item, ensure_ascii=False)
      f.write(dic)
      f.write("\n")

# Read data
def read_data(path):
  data = []  # pylint: disable=redefined-outer-name
  with open(path, "r") as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):  # pylint: disable=unused-variable
      dic = json.loads(line)
      data.append(dic)
  #print("data_length:" + str(len(data)))
  return data

def split_data(data, train_ratio, valid_ratio, folder_labels, rand_seed):  # pylint: disable=redefined-outer-name
  """Splits data into train, validation and test according to ratio."""
  train_data = []  # pylint: disable=redefined-outer-name
  valid_data = []  # pylint: disable=redefined-outer-name
  test_data = []  # pylint: disable=redefined-outer-name
  
  # save the data number of label as dict
  num_dic = {}
  for val in folder_labels:
     val = val.strip( ',' )
     num_dic[val] = 0
  num_dic["negative"] = 0
  
  for idx, item in enumerate(data):  # pylint: disable=unused-variable
    for i in num_dic:
      if item["gesture"] == i:
        num_dic[i] += 1
  print("num_dic: {}".format(num_dic))
  
  # each label should have same ratio for balanced training
  train_num_dic = {}
  valid_num_dic = {}
  for i in num_dic:
    train_num_dic[i] = int(train_ratio * num_dic[i])
    valid_num_dic[i] = int(valid_ratio * num_dic[i])
  random.seed(rand_seed)
  random.shuffle(data)
  
  for idx, item in enumerate(data):
    for i in num_dic:
      if item["gesture"] == i:
        if train_num_dic[i] > 0:
          train_data.append(item)
          train_num_dic[i] -= 1
        elif valid_num_dic[i] > 0:
          valid_data.append(item)
          valid_num_dic[i] -= 1
        else:
          test_data.append(item)
  print("train_length: " + str(len(train_data)))
  print("valid_length: " + str(len(valid_data)))
  print("test_length: " + str(len(test_data)))
  return train_data, valid_data, test_data

if __name__ == "__main__":
  data = []  # pylint: disable=redefined-outer-name
  
  parser = argparse.ArgumentParser()
  parser.add_argument(
        '--folders',
        type=str,
        nargs="+",
        default=["ring", "slope", "wing"],
        help='Read data from folders, ex: "/slope", "/ring"')
  parser.add_argument(
        '--names',
        type=str,
        nargs="+",
        default=["hyw", "shiyun", "tangsy", "dengyl", "zhangxy", "pengxl", "liucx", "jiangyh", "xunkai"],
        help='Person name')
  parser.add_argument(
        '--out_dir',
        type=str,
        default='out_dataset_1',
        help='What model architecture to use')
  parser.add_argument(
        '--seq_length',
        type=int,
        default=128,
        help='Decide the feature number of 1 dim')
  parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.6,
        help='train ratio of dataset',)
  parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='valid ratio of dataset',)
  parser.add_argument(
        '--rand_seed',
        type=int,
        default=30,
        help='random seed which is a fix random list',)
  parser.add_argument(
        '--neg_data_num',
        type=int,
        default=100,
        help='generate how many negative data',)
  FLAGS, _ = parser.parse_known_args()
  
  # user collecting normal data folders
  for idx1, folder in enumerate(FLAGS.folders):
    folder = folder.strip( ',' )
    for idx2, name in enumerate(FLAGS.names):
      name = name.strip( ',' )
      raw_file = "./%s/%s/output_%s_%s.txt" % ("data", folder, folder, name)
      if os.path.exists(raw_file):
        print("raw data folder: {%s}" % (raw_file))
        prepare_original_data(folder, name, data, raw_file, FLAGS.seq_length)
      else:
        print("raw data folder: {%s} doesn't exist! <Please notice the balance of training data.>" % (raw_file))  
  
  # user collecting negative data folders. output_negative_1, output_negative_2, ... output_negative_5    
  for idx in range(1):
    prepare_original_data("negative", "negative%d" % (idx + 1), data,
                          "./%s/negative/output_negative_%d.txt" % ("data", (idx + 1)), FLAGS.seq_length)
  
  # auto generated negative data   
  generate_negative_data(data, FLAGS.seq_length, FLAGS.neg_data_num, FLAGS.train_ratio, FLAGS.val_ratio)
  
  print("data_length: " + str(len(data)))
  if not os.path.exists("./data/%s" % (FLAGS.out_dir)):
    os.makedirs("./data/%s" % (FLAGS.out_dir))
  write_data(data, "./data/%s/complete_data" % (FLAGS.out_dir))
  
  # split the data
  data = read_data("./data/%s/complete_data" % (FLAGS.out_dir))
  train_data, valid_data, test_data = split_data(data, FLAGS.train_ratio, FLAGS.val_ratio, FLAGS.folders, FLAGS.rand_seed)
  write_data(train_data, "./data/%s/train" % (FLAGS.out_dir))
  write_data(valid_data, "./data/%s/valid" % (FLAGS.out_dir))
  write_data(test_data, "./data/%s/test" % (FLAGS.out_dir))

