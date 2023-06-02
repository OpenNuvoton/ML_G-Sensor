# G-sensor_Example
- Gesture Recognition Magic Wand Training Scripts
- The scripts in this directory can be used to train a TensorFlow model that
classifies gestures based on accelerometer data. The code uses
TensorFlow 2.0. The resulting model is less than 20KB in size.
This project was inspired by the [Gesture Recognition Magic Wand](https://github.com/jewang/gesture-demo)
project by Jennifer Wang.

---
## 1. First step
### 1. Install virtual env
- If you havn't install [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise), please follow the steps to install python virtual env and ***choose `NuEdgeWise_env`***.
- Skip if you have done.
### 2. Running
- The `magic_wand_start.ipynb` will help you prepare data, train the model, and finally convert to tflite & c++ file.

## 2. Work Flow
### 1. Dataset
- The dataset is from 3-dims x, ,y, z accelerometer of different gesture. 
- User can collect their own data by running this code [m467 sensor_collect](https://github.com/stanlytw/M467_MAGICWAND/tree/main/SampleCode/numaker_IoT_m467_sensor_collect) on m467 EVB.
- `magic_wand_start.ipynb` will help you collect and prepare the data.
### 2. Training
- Use `magic_wand_start.ipynb` to train the model at local.

#### Training in Colab
- Use `magic_wand_colab.ipynb` to upload the dataset and train on the colad.

### 3. Test & Deployment
- Use `magic_wand_start.ipynb` to convert to tflite & c++ file.

## 3. Inference code
- [m467 g-sensor_magicwand](https://github.com/OpenNuvoton/ML_M460_SampleCode/tree/master/SampleCode/NuMagicWand)





