# AI Egyptian Dialect Fake Audio Detection

## Overview

This project focuses on leveraging machine learning techniques to detect fake audio samples in Egyptian Dialect. The primary approach involves using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models, which are well-suited for sequential data like audio signals. The models are trained on a dataset consisting of both genuine and spoofed audio samples, aiming to classify whether an audio clip is authentic or manipulated.

## Motivation

With the proliferation of digital manipulation tools, the ability to verify the authenticity of audio recordings is crucial. This project addresses the specific challenge of detecting fake audio in Egyptian Dialect, which is essential for maintaining the integrity of audio-based content in various applications.

## Project Structure

The project is structured into several key components:

### 1. Data Preparation

- **Audio Processing**: Utilizes the LibROSA library for audio loading, preprocessing, and feature extraction.
- **Feature Extraction**: Extracts Mel-Frequency Cepstral Coefficients (MFCC) from audio signals, a widely used feature representation for audio analysis.
- **Dataset Organization**: Prepares the dataset by categorizing audio files into genuine and spoofed categories, ensuring balanced representation for model training.

### 2. Model Training

- **LSTM and GRU Models**: Constructs sequential models using TensorFlow/Keras, incorporating LSTM and GRU layers known for their ability to capture long-term dependencies in time-series data.
- **Training Configuration**: Defines model architecture, optimizers (e.g., Adam), and callbacks (e.g., Early Stopping) to monitor and optimize training performance.
- **Hyperparameter Tuning**: Provides flexibility to adjust hyperparameters such as learning rate, batch size, and number of epochs based on experimentation and validation results.

### 3. Evaluation

- **Performance Metrics**: Evaluates model performance using standard metrics such as accuracy, loss curves, confusion matrices, and classification reports.
- **Visualization**: Visualizes training and validation metrics to assess model convergence and identify potential overfitting or underfitting issues.
- **Model Interpretation**: Generates insights into model predictions through visualization of audio spectrograms, highlighting differences between genuine and spoofed audio characteristics.

### 4. Deployment Considerations

- **Scalability**: Discusses considerations for scaling the model to handle larger datasets or real-time inference scenarios.
- **Integration**: Provides guidelines for integrating the trained models into production environments or deploying as part of a larger audio verification system.
- **Performance Optimization**: Strategies for optimizing model inference speed and memory footprint without compromising detection accuracy.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy
- pandas
- librosa
- matplotlib
- scikit-learn

## Usage

1. **Data Preparation**: Ensure audio data is structured and labeled correctly. Use scripts provided to preprocess audio files into MFCC features stored in `data.json`.
2. **Model Training**: Customize model architecture and training parameters in `train_model.py`. Run the script to train LSTM and GRU models on the prepared dataset.

3. **Evaluation**: Evaluate model performance using metrics and visualizations provided in `evaluate_model.py`. Fine-tune models based on evaluation results to improve detection accuracy.

## Files and Directories

- `audio-for-lstm/`: Directory containing audio files for training and testing.
- `train_model.py`: Script for configuring and training LSTM and GRU models.
- `evaluate_model.py`: Script for evaluating model performance and generating evaluation metrics.
- `data.json`: JSON file storing preprocessed MFCC features and corresponding labels for model training.

## References

- [LibROSA Documentation](https://librosa.org/doc/latest/index.html): Documentation for the audio processing library used in this project.
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras): Official TensorFlow/Keras API documentation for building and training deep learning models.
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html): Documentation for machine learning tools used for evaluation and metrics computation.

## Authors

- Saif Eldeen Khaled Emera, Ziad Tarek

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for more details.
