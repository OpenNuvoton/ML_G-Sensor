# G-sensor_Example
- Gesture Recognition Magic Wand Training Scripts.
- The scripts in this directory can be utilized to train a TensorFlow model that classifies gestures using accelerometer data. The code is designed for TensorFlow 2.0. The resulting model has a size of less than 20KB.
This project was inspired by the [Gesture Recognition Magic Wand](https://github.com/jewang/gesture-demo)
project by Jennifer Wang.

---
## 1. First step
### 1. Install virtual env
- If you haven't installed [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow these steps to install Python virtual environment and ***choose `NuEdgeWise_env`***.
- Skip if you have already done it.
### 2. Running
- The `magic_wand_start.ipynb` notebook will help you prepare data, train the model, and finally convert it to a TFLite and C++ file.

## 2. Work Flow
### 1. Dataset
- The dataset consists of accelerometer readings in three dimensions: x, y, and z, collected from various gestures. 
- Users can collect their own data by running the code [m467 sensor_collect](https://github.com/OpenNuvoton/ML_M460_SampleCode/tree/master/SampleCode/numaker_IoT_m467_sensor_collect) on the m467 EVB.
- `magic_wand_start.ipynb` will assist you in collecting and preparing the data.
### 2. Training
- Use `magic_wand_start.ipynb` to train the model locally.

#### Training in Colab
- Utilize `magic_wand_colab.ipynb` to upload the dataset and train on Google Colab.

### 3. Test & Deployment
- Use `magic_wand_start.ipynb` to convert to TFLite and C++ files.

## 3. Inference code
- [ML_M460_SampleCode](https://github.com/OpenNuvoton/ML_M460_SampleCode)
   - `tflu_magicwand`
   - `tflu_magicwand_sensor_collect`





