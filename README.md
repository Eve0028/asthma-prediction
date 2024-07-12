# Asthma prediction

Asthma prediction based on breathing sound using three pre-trained convolutional networks: VGGish, ResNet50, DenseNet121.</br>
The project was done for a graduate engineering thesis.

## Introduction 

The sound of breathing recorded at the patient's mouth contains a lot of information that can be used to diagnose respiratory diseases. The project seeks to test whether the presence of asthma can be diagnosed using it. The model was trained on a sizable database of sound data (4,774) collected from asthmatics (2,387) and people without the disease (2,387). The sound files were recorded on users' equipment - [see datasource](#data-source).

## Data Configuration

To set up the project data, you need to add a few files that are not included in the repository due to size or confidentiality reasons. Instructions below:

1. **data_params.py.temp** (in prediction/data/)
    - This file contains two parameters: `CSV_DATA_SAMPLES` and `DATA_SAMPLES`.
    - Modify the file name to `data_params.py`, removing `.temp` at the end.

2. **CSV_DATA_SAMPLES**
    - Add a CSV file containing data information of the samples for training the model.
    - Ensure that the CSV file has the following fields related to the data in the 'DATA_SAMPLES' folder:
        ```
        Uid;Folder Name;Breath filename;label;split
        ```
        - Set the path to this file in the `CSV_DATA_SAMPLES` variable in the `data_params.py` file.

3. **DATA_SAMPLES**
    - Add a file containing recordings of samples for training the model.
    - Set the path to this file in the `DATA_SAMPLES` variable in the `data_params.py` file.

4. **vggish_model.ckpt**
    - Add the file with weights `vggish_model.ckpt` for the [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model to the 'prediction/vggish' or/and 'prediction/my_vggish_model' folder.

## Data Source

This project was developed using data from the University of Cambridge (project on COVID-19 diagnosis using cough, breath, and speech recordings). You can find the repository [here](https://github.com/cam-mobsys/covid19-sounds-neurips).

In addition to the recordings, patient data included additional information such as:
- demographics, medical history, symptoms;
- metadata such as age, gender, presence of comorbidities (e.g., asthma, COPD, pulmonary fibrosis) and smoking.

Distribution of users in the asthmatic group.</br>
![asthmatic_users.png](images%2Fasthmatic_users.png)

## Recording filtration

There were several recordings of the same patient in the data set (the development of the disease was examined over several days). It was decided to take only one sample from such a set. The main reasons are:
- preventing a more favorable classification of the patient into a specific class (due to, for example, a larger number of his recordings in the training set);
- spreading samples from the same user in the training and validation set at the same time - would lead to data leakage.

The dataset came from many platforms (browser application, Android and iOS) - therefore the recordings had different types of extensions (.ogg, .m4a, .wav and .webm) and different sampling rates. All files have been converted to the lossless .wav format (however, this will not change the 'less accurate' data with the .m4a extension).

It was decided that samples with a sampling rate less than 16 kHz would be discarded. The rest of the recordings were converted to 16 kHz.
</br>Number of all recordings after initial filtering: 28,034.
</br>Number of asthmatics: **2,387 (approx. 8.5%)**.

### First attempt 

Due to the running out of time and the impending deadline for my thesis - I decided to use under-sampling to deal with unbalanced classes (fewer samples -> shorter learning time). 

Balanced class distribution was applied to the training, validation and test set. However, after proper study of the problem - balanced class distribution should be applied only in the training set. The validation and test set should keep the 'natural' distribution of classes (sick/healthy) at a ratio of 8:92 - to make the model results more reliable. 

After initial training, the models did not perform very well. With the deadline approaching faster and faster and with poor results - it was decided to proceed with a second attempt. 

### Second attempt 

People without asthma sometimes have additional symptoms and diseases that affect their breathing sound - making it difficult to classify them.
At this point, the reverse possibility was also realized - of asthmatics showing no respiratory distress.
Probably the model was not able to find any features indicative of asthma, because the patients (at the time of the sample donation) were under treatment for asthma - which dulls its symptoms and its effect on breathing sound to a minimum!
If the asthmatic had other diseases affecting the breath sound - the model probably mistook their features for asthma.

Despite this, it was decided not to give up and try once again to filter the data. This time, the dataset was cleared additionally of individuals:
- having the given symptoms: dry cough, wet cough, sore throat, runny or blocked nose, chest tightness, loss of taste or smell, difficulty breathing;
- having diseases that affect the respiratory capacity of the lungs: angina, cancer, emphysema / COPD (chronic obstructive pulmonary disease), cystic fibrosis, pulmonary fibrosis, other lung diseases;
- compulsive smokers (daily) and former smokers.

The collection has been significantly reduced - **diseased class: 682**. 
With such a small sample, the risk of early network overfitting increases significantly. 
When tuning model parameters, a smaller number of trained last layers of the base model and smaller number of neurons for the FC ReLU layer were checked.
A larger dropout and L2 regularization were also tested.</br>
Such narrowing of the data also limits the deployability of the model and its use to only those who meet the above criteria. 
**Always train the network on data as close as possible to the data for which the model will be designed.** 

As I expected - the results did not improve significantly - reaching less than 60% accuracy and AUC (and an even lower f1-score) -ᴖ-.</br>
Nevertheless (hoping that the calculations are correct) I have at least partially proved that asthma treatment is effective! :D 

## Preprocessing
The networks were trained on log mel spectrograms, which represent the amplitude of a signal in the time and frequency domains.
Audio conversion to the model is as follows: 
- the file is sampled to 16 kHz mono and scaled to values in the range [-1; 1]; 
- the spectrogram is calculated using STFT with a window size of 25 ms, a step equal to 10 ms, and an overlay of the Hann window; 
- the spectrogram is converted to mel scale using 64 filters covering a frequency range of 125-7500 Hz; 
- this is followed by logarithmization of the amplitude - log(mel-spectrum + 0.01) - where an offset equal to 0.01 is used to avoid a logarithm equal to zero. 

The respiratory recordings have an average length of a few to several tens of seconds. 
The entire spectrogram has been divided into smaller sections.
A 96x64 spectrograms (which includes 64 mel bands and 96 frames of 10 ms each) is transmitted to the model input 
(to match the data format on which the VGGish network was pre-trained).
The last of these ('sub-sample') if too short - is omitted.

## Architecture
The structure of the models comes from a paper called *COVID-19 Sounds: A Large-Scale Audio Dataset for Digital Respiratory Screening*. 
Instead of using three different types of recordings (cough, breath and voice) - only breath was used for asthma detection in the current project.</br>
All the models tested have the same architecture with a different base model.</br>
![model_architecture.jpg](images%2Fmodel_architecture.jpg)

From a single breath sample, a few dozen `sub-samples` (96x64 spectrogram) of data are produced for input to the model.
The model makes one prediction from all `sub-samples` from one patient - by pooling all extracted features through the base model before sending them to the Dense layers (for prediction). 

## Training

### Hyper-parameters

#### Applied to all models in advance
- optimizer: Adam,
- loss function: categorical cross-entropy.

#### Tuned:
- base learning rate - for the base (pre-trained) model,
- upper learning rate - for attached fully connected classification layers,
- learning rate decay,
- L2 regularization,
- dropout,
- number of neurons for the FC ReLU layer,
- number of trained layers (in the base model).

### Training Pipeline
1) Importing the weights of the pre-trained base model (+ adding additional Dense layers with a classification layer). Adding `training=False` so that the batchnorm layers will not update their batch statistics during fine-tuning.
2) Freezing the layers of the base model (adding `training=False` beforehand does the trick).
3) Training the last new Dense layers for several epochs (convergence).
4) Unfreezing the appropriate number of layers of the base model, counting the layers from the end of the model.
5) Fine-tuning the model.

### Evaluation metrics
- Accuracy,
- TPR, TNR,
- F1-Score,
- AUC,

## Models
Folder structure in `prediction` directory.
### vggish
VGGish model - files can be found [here](https://github.com/tensorflow/models/tree/master/research/audioset/vggish).</br>
`vggish_slim.py`: Model definition. The base for the models in the `model` and `my_vggish_model` folders.</br>
`vggish_params.py`: Hyperparameters.</br>
`vggish_input.py`: Converter from audio waveform into input examples.</br>
`mel_features.py`: Audio feature extraction helpers.</br>

### model
The VGGish model was used as the base model and feature extractor.
Modified code from [original repo](https://github.com/cam-mobsys/covid19-sounds-neurips):
- originally a TensorFlow1 was used - modification to properly fires on the TensorFlow2 version (tf.compat.v1, etc.).
- modification of the model architecture to accept a sample type with one recording/modality (breath) instead of three.

### my_image_model
The architecture is based on the ResNet50 and DenseNet121 models. 
A custom training loop was used.
</br>The ` tensorflow_addons` library used in the code is no longer supported. 

#### Additional samples preprocessing
The sample audio was converted to log mel spectrogram form in the same way as for the VGGish network (except for the amount of filters applied when converting to mel scale). The values in the array were rescaled to be within the 8-bit range (0-255). 

The ResNet50 and DenseNet121 networks were trained on RGB (3-channel) images with dimensions (height x width x number_channels) - this means that their pre-trained weights cannot be implemented into a network that accepts a single-channel data format. 
The conversion of a grayscale log mel spectrogram image to an image having 3 channels (same values on each channel) was used. 

To ensure that the images have the same format and dimensions as the data used to pre-train the base model (224x224) - 128 mel filters and interpolation to a height equal to 244 pixels were applied. 

The images are not saved to computer disk (all preprocessing is included in the training pipeline) - but it is worth mentioning that their eventual saving should be done to lossless PNG format. 

Finally, the spectrogram image is divided (by 'length') into 'sub-images' of size: 244x244.

### my_vggish_model
The VGGish model was used as the base model and feature extractor.

This model is a 'correctly' reworked version of the model from the `model` folder to TwnsorFlow2. 
This version tested **averaging the results after the softmax** (classification) layer instead of before the Dense layers.
</br>This model is the only one that has not been tested on a dataset of breath recordings. It was added after the engineering work was finished - it hasn't had a chance to be tested yet :c. 
</br></br>
Functionality used:
- functional API, 
- Model.fit API - instead of custom training loop, 
- addition of TimeDistributed layers - also for Dense layers, 
- tf.data API: 
  - use of generator to create samples of different sizes in batch (different number of 'sub-samples' in one signal), 
  - definition of sample in batch as [None, sample_width, sample_height] - so that the graph does not require retracing. 
  - use of prefetch - step time reduction. 

## Worth adding 
- Frequency filtering of recordings (breathing sounds heard from the patient's mouth have a frequency range: 200-2000 Hz).
- Using all samples from healthy subjects, rather than discarding them in favor of matching their number to the minority class size - and assigning them weights accordingly to reduce the effect of unbalanced classes (sample weighting in the loss function).
- Distributing the classes (sick/healthy) in an 8:92 ratio in the validation and test sets, or using cross-validation (and stratified sampling).
- Equal distribution of patients into the appropriate classes taking into account the same: age range, gender, language (although it may not necessarily matter much for breath sounds) and other additional patient information. Which will minimize the impact of bias on the results. 
