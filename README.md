# Audio Classification using LSTM

## Please note that the full project is not currently uploaded; for now, it's only the HTML files for both GRU and LSTM models. The whole project will be uploaded soon. Thanks!

This project implements a deep learning model using Long Short-Term Memory (LSTM) networks for audio classification. It focuses on distinguishing between genuine and spoofed audio samples of Egyptian dialect.

## Overview

This repository contains scripts and data for training and evaluating an LSTM model on audio data. The dataset used is collected data of Egyptian dialect, including both genuine and spoofed audio recordings.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Librosa
- NumPy
- Matplotlib
- Pandas

## Dataset

The dataset used for training includes collected data of Egyptian dialect, consisting of genuine and spoofed audio recordings.

## Project Structure

audio-for-lstm/
│
├── README.md # This file, project overview and setup
├── data.json # JSON file containing MFCC data
├── lstm_model.py # Script for building and training LSTM model
├── preprocess_data.py # Script for preprocessing audio data
├── requirements.txt # Python dependencies
├── audio/
│ ├── 0/ # Folder for genuine audio files
│ └── 1/ # Folder for spoofed audio files
└── archive/
└── ... # Contains collected data of Egyptian dialect


## Usage

1. **Data Preprocessing**: Use `preprocess_data.py` to preprocess audio files into MFCC format and store them in `data.json`.

2. **Model Training**: Run `lstm_model.py` to train the LSTM model using the preprocessed data.

3. **Evaluate Model**: Evaluate the trained model on test data and visualize results using the scripts provided.

## Running the Code

1. Clone this repository:

git clone [https://github.com/your-username/your-repo.git](https://github.com/SaifEKhaled/DetectingFakeAudioOfEgyptianDialect)
cd your-repo

