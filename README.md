# Deepfake Video Detection Using CNNs and Explainable AI

A group project that detects deepfake videos by combining CNNs (for what's in each frame) with LSTMs (for how frames change over time), compares them with simpler "glass-box" models, and uses Grad-CAM++, saliency maps and model coefficients to explain the predictions. It also looks at how the system lines up with the EU AI Act and GDPR.

Built as a combined project for two modules of the MSc in Human Centered Artificial Intelligence at Technological University Dublin (2025).

## Team

- Christy Maria Joseph
- Neetha Mariam Sam
- Praveen Sivakumar
- Vibhooshan Gunasekaran

### My part

I built, trained and tested both black-box models: the baseline CNN + LSTM, and the MobileNetV2 + LSTM transfer learning model that became our best model. I also worked on the explainability side, putting Grad-CAM++ and saliency maps on the MobileNetV2 + LSTM model, and helped with the coefficient analysis for the logistic regression model. In the report, I wrote parts of the black-box model and XAI sections and helped with the methodology, references and future work. My full write-up is in `Praveen_Sivakumar_X00230952_Individual_Contribution.docx`.

## Results at a glance

| Model | Type | Test accuracy |
|---|---|---|
| CNN + LSTM (baseline) | Black box | 73% |
| **MobileNetV2 + LSTM** | Black box | **96%** |
| Logistic Regression on MobileNetV2 features | Glass box | 96% |
| Decision Tree on MobileNetV2 features | Glass box | 82% |

All results are on a held-out test set of 90 videos (45 real, 45 fake). MobileNetV2 + LSTM was picked as the final model because it models the changes between frames as well as the frames themselves, and its Grad-CAM++ and saliency maps were easier to interpret than the logistic regression coefficients.

## Research questions

1. How well can a CNN + LSTM architecture tell real videos from deepfakes?
2. Which XAI techniques (Grad-CAM++, saliency maps for the black-box model, feature coefficients for the glass-box model) best show what drives the model's decisions?
3. How far do these explanations meet EU AI Act transparency and GDPR auditability requirements?

## Dataset

[Deep Fake Detection (DFD) Entire Original Dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset) on Kaggle

- 364 real videos of 28 actors in 16 different scenes
- 3,068 fake videos made with face-swapping
- All videos are 1920×1080, mostly 400 to 1,000 frames long
- Heavily imbalanced, with about 8 fake videos for every real one

## Method

**1. Preprocessing**
- Picked 300 real and 300 fake videos at random to balance the classes and keep training manageable
- Took 10 evenly spaced frames from each video, so 6,000 frames in total
- Resized frames to 128×128, converted them from BGR to RGB and scaled pixels to 0 to 1
- Stratified split: 70% train (420 videos), 15% validation (90), 15% test (90)

**2. Baseline: CNN + LSTM**
- Three `TimeDistributed` conv blocks (32, 64, 128 filters), each with BatchNorm, MaxPooling and Dropout 0.25
- Global average pooling per frame, then LSTM(256) and LSTM(128) with Dropout 0.3
- Sigmoid output for real (0) or fake (1)
- Adam (lr = 1e-4), binary cross-entropy, batch size 8, up to 20 epochs with EarlyStopping and ModelCheckpoint

**3. Transfer learning: MobileNetV2 + LSTM**
- A frozen MobileNetV2 (ImageNet weights) wrapped in `TimeDistributed` extracts features from each frame
- The same LSTM layers and output as the baseline
- Up to 30 epochs with EarlyStopping (patience 5) and ModelCheckpoint

**4. Glass-box models**

MobileNetV2 turns each frame into a 1,280-length feature vector, and the vectors are averaged into one per video. A Logistic Regression and a Decision Tree are then trained on these vectors.

**5. Explainability**
- **Grad-CAM++** heatmaps from MobileNetV2's last conv layer
- **Saliency maps** from input gradients
- **Logistic regression coefficients** to see which features push a prediction towards real or fake

## Results

### CNN + LSTM (baseline)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Real | 0.68 | 0.87 | 0.76 |
| Fake | 0.82 | 0.60 | 0.69 |
| **Accuracy** | | | **0.73** |

Training accuracy kept going up, but validation accuracy jumped around between about 45% and 72%, a clear sign of overfitting. The model also missed 40% of the fake videos.

### MobileNetV2 + LSTM (best model)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Real | 1.00 | 0.91 | 0.95 |
| Fake | 0.92 | 1.00 | 0.96 |
| **Accuracy** | | | **0.96** |

Training and validation accuracy both passed 95% within a few epochs and stayed close together. The model caught every fake video in the test set (45 of 45) and got 41 of 45 real videos right.

### Glass-box models

| Model | Real precision / recall | Fake precision / recall | Accuracy |
|---|---|---|---|
| Logistic Regression | 1.00 / 0.91 | 0.92 / 1.00 | 0.96 |
| Decision Tree | 0.94 / 0.69 | 0.75 / 0.96 | 0.82 |

The logistic regression matched the deep model, which shows how much of the work the pretrained MobileNetV2 features were doing. The decision tree leaned towards calling real videos fake, and with 1,280 features it grew too complex to be easy to read.

### What the explanations showed

- **Grad-CAM++ and saliency maps:** on correct predictions, the model focused on the person, especially the face, hands, upper body and gestures. On wrong predictions, its attention drifted to the background, such as desks and walls. Each method was clearer on different frames, so we found them more useful together than apart.
- **Coefficients:** they show which features push a prediction towards real or fake. But the features come from MobileNetV2, so a feature like `feat_1` doesn't match anything a person can name, like "eyes" or "mouth". For this project, the visual explanations of the black-box model turned out to be more useful than the glass-box coefficients.

## Regulation and ethics

The report maps the system against:

- **GDPR:** data minimisation (only 10 frames per video), a public research dataset, removing metadata, and support for the right to explanation through XAI
- **EU AI Act:** transparency (Art. 13), human oversight as a decision-support tool rather than an automated decision-maker (Art. 14), accuracy and reliability (Art. 15), and documentation (Arts. 16 to 18)
- **UN SDGs:** 9 (industry and innovation), 16 (peace, justice and strong institutions) and 17 (partnerships)

## Future work

- Larger and more varied datasets, like low-light footage or partly hidden faces
- Attention, transformer or 3D CNN architectures
- Detection that uses both audio and video
- Making the model hold up better against new deepfake methods
- Getting the models ready for real use in social media moderation or forensics

## Tech stack

Python, TensorFlow/Keras, MobileNetV2, LSTM, scikit-learn, OpenCV, NumPy, Matplotlib, Seaborn, Grad-CAM++, saliency maps

## Files

- `Praveen_Sivakumar_X00230952_Notebook.ipynb`: the full notebook, covering data exploration, frame extraction, all four models, evaluation and XAI
- `Praveen_Sivakumar_X00230952_HTML.html`: an HTML export of the notebook with all outputs, for viewing without Jupyter
- `Praveen_Sivakumar_X00230952_Report.docx`: the group report
- `HCDL-Combined_Project-PPT.pptx`: the group presentation slides
- `Praveen_Sivakumar_X00230952_Individual_Contribution.docx`: my individual contribution statement
