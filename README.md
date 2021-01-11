# Face-Verification-through-Classification-And-Metric-Learning
## Introduction
This project uses Resnet18 to perform face verification. Resnet18 learns to extract face embeddings during training, and cosine similarities between embeddings are calculated to measure face similarities. The verification performance is evaluated through AUC score.
## Dataset
The dataset contains images for 4000 different person. Each person has 20-100 sample images.
## Training
The model was pretrained through N-way classification. After 10 epoch, it was able to reach 66% classification and 90% AUC score.


