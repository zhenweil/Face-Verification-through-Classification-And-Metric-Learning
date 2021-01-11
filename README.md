# Face-Verification-through-Classification-And-Metric-Learning
## Introduction
This project uses Resnet18 to perform face verification. Resnet18 learns to extract face embeddings during training, and cosine similarities between embeddings are calculated to measure face similarities. The verification performance is evaluated through AUC score.
## Dataset
The dataset contains images for 4000 different person. Each person has 20-100 sample images.
## Training
The model was pretrained through N-way classification. After 10 epoch, it was able to reach 66% classification accuracy and 90% AUC score. 
The model was then fine-tuned through metric learning. During metric learning, three images were fed into the model: an anchor image, a positve image (image that belongs to the same person as the anchor image), and a negative image (iamge that belongs to a different person as the anchor image). The positive image and negative image were randomly selected from training samples. The model extract embeddings from these images and triplet loss was calculated. The formula for triplet loss is shown below: 
```
L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```


