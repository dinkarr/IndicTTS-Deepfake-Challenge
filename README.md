# IndicTTS-Deepfake-Challenge

The IndicTTS-Deepfake-Challenge project focuses on detecting deepfake Text-to-Speech (TTS) audio using deep learning techniques. The goal of the project is to develop a model capable of distinguishing between real and synthetic (deepfake) TTS audio based on features extracted from the audio signals. The model is built upon a pre-trained transformer-based architecture and is finetuned for the specific task of deepfake detection.

## Objective

- Detect deepfake TTS audio by fine-tuning a pre-trained model.
- Utilize a feature extractor to preprocess and transform the raw audio into input features.
- Train the model using a custom data collator to handle padding and batching for variable-length audio samples.
- Evaluate model performance using metrics such as accuracy, F1 score, and AUC-ROC.

## Data Processing and Feature Extraction

### Audio Processing
- The audio input is processed to ensure uniform length by padding or truncating to a maximum length of 16,000 samples.
- The feature extractor is applied to convert the raw audio to input features suitable for the model.
- The processed audio features are then passed through the model for classification, with labels representing whether the audio is synthetic (`is_tts`).

### Data Augmentation
- **Padding and Truncation**: If the audio length is shorter than the maximum length (16,000 samples), it is padded; otherwise, it is truncated to fit.
- **Feature Extraction**: The feature extractor processes the audio and returns the relevant features for training.

## Data Collator
A custom `DataCollatorWithPadding` class is implemented to:
- Apply padding to the input features using the feature extractor.
- Convert labels to tensors, ensuring they are consistent and ready for model training.

## Model Fine-Tuning

- **Pre-trained Model**: A pre-trained transformer model is fine-tuned for the task of deepfake detection.
- **Model Freezing**: During the fine-tuning process, the feature encoder is frozen to prevent its weights from updating. Only the final layers are trained.
- **Model Head**: The model head is modified to output the desired classification (real vs. synthetic audio).
- **Device Handling**: The model is moved to the GPU for faster training using `.to("cuda")` if a GPU is available.

## Evaluation Metrics

The model is evaluated using the following metrics:
- **Accuracy**: The percentage of correctly classified samples.
- **F1 Score**: The harmonic mean of precision and recall.
- **ROC AUC**: Area under the ROC curve for assessing the model's ability to distinguish between classes.

### Training Progress:
Snapshot of the training and evaluation metrics:

| Step   | Training Loss | Validation Loss | Accuracy | F1    | ROC AUC |
|--------|---------------|-----------------|----------|-------|---------|
| 500    | 0.5999        | 0.4321          | 0.8091   | 0.7741 | 0.9567  |
| 1000   | 0.3113        | 0.3181          | 0.9196   | 0.9158 | 0.9872  |
| 1500   | 0.2389        | 0.1524          | 0.9656   | 0.9658 | 0.9941  |
| 2000   | 0.1656        | 0.0498          | 0.9846   | 0.9849 | 0.9987  |
| 2500   | 0.1267        | 0.1810          | 0.9569   | 0.9565 | 0.9986  |
| 3000   | 0.0790        | 0.0799          | 0.9810   | 0.9813 | 0.9987  |

### Final Training Output:

TrainOutput(global_step=3499, training_loss=0.22596901203775854, metrics={'train_runtime': 1745.6619, 'train_samples_per_second': 16.035, 'train_steps_per_second': 2.004, 'total_flos': 8.48358162602304e+17, 'train_loss': 0.22596901203775854, 'epoch': 1.0})

## Model Inference

Once trained, the model can be used to predict the likelihood of an audio sample being synthetic. The process involves:

1. Extracting features from the audio.
2. Feeding the features into the model.
3. Using the softmax function to compute probabilities for each class (real or synthetic).
4. Storing the prediction results in a DataFrame for analysis.

## Training Configuration

- **Training Arguments**:
  - **Batch Size**: 8 samples per device for both training and evaluation.
  - **Epochs**: The model is trained for 1 epoch.
  - **Evaluation Strategy**: Evaluates the model at every 500 steps.
  - **Learning Rate**: Set to 1e-4, with a warmup period of 1000 steps.
  - **Saving Models**: The best model is saved after every 1000 steps, with a total of 3 models being kept at any time.

## Conclusion

The IndicTTS-Deepfake-Challenge is a comprehensive approach to deepfake TTS detection, utilizing a pre-trained transformer model fine-tuned on an audio classification task. By leveraging robust feature extraction, data augmentation, and detailed model evaluation, the model achieves high performance in distinguishing synthetic speech from real human speech.
