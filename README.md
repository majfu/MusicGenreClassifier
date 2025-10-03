My first bigger machine learning project, I am very proud of it (˶ᵔ ᵕ ᵔ˶)

I trained models for music genre classification task. The first approach was a single model for multilabel prediction, but the performance was mediocre. The second approach was to train a binary classifier for each genre, and here the performance improved. Combining the models predictions seems to yield even better results.

There is also a frontend written in React (my first time working with React) and a Falsk backend to process the audio files.


Results of model evaluation on test dataset

Multilabel:
Test Loss: 0.2805
Macro F1 Score: 0.6073

Multilabel per-genre:
Classical: F1=0.8592, Precision=0.7625, Recall=0.9839, Accuracy=0.9682
Dance: F1=0.3514, Precision=0.5417, Recall=0.2600, Accuracy=0.9237
Electronic: F1=0.6087, Precision=0.7179, Recall=0.5283, Accuracy=0.8855
Experimental: F1=0.6010, Precision=0.6237, Recall=0.5800, Accuracy=0.8776
Hip-Hop: F1=0.7059, Precision=0.8571, Recall=0.6000, Accuracy=0.9205
Metal: F1=0.7059, Precision=0.6923, Recall=0.7200, Accuracy=0.9523
Old-Time: F1=0.9903, Precision=0.9808, Recall=1.0000, Accuracy=0.9984
Pop: F1=0.3648, Precision=0.4915, Recall=0.2900, Accuracy=0.8394
Punk: F1=0.3256, Precision=0.3889, Recall=0.2800, Accuracy=0.9078
Rock: F1=0.7945, Precision=0.7982, Recall=0.7909, Accuracy=0.9285
Techno: F1=0.3733, Precision=0.5600, Recall=0.2800, Accuracy=0.9253


Binary model for each genre:
Classical: F1=0.9381, Precision=0.9138, Recall=0.9636, Accuracy=0.9364, Loss=0.2948
Dance: F1=0.7708, Precision=0.7255, Recall=0.8222, Accuracy=0.7556, Loss=0.5850
Electronic: F1=0.8732, Precision=0.8732, Recall=0.8732, Accuracy=0.8732, Loss=0.3014
Experimental: F1=0.7892, Precision=0.7816, Recall=0.7970, Accuracy=0.7871, Loss=0.4308
Hip-Hop: F1=0.9119, Precision=0.9312, Recall=0.8934, Accuracy=0.9137, Loss=0.2474
Metal: F1=0.7925, Precision=0.7778, Recall=0.8077, Accuracy=0.7885, Loss=0.3995
Old-Time: F1=0.9778, Precision=0.9778, Recall=0.9778, Accuracy=0.9778, Loss=0.0321
Pop: F1=0.7081, Precision=0.7184, Recall=0.6981, Accuracy=0.7123, Loss=0.5730
Punk: F1=0.8403, Precision=0.8018, Recall=0.8826, Accuracy=0.8322, Loss=0.3996
Rock: F1=0.8880, Precision=0.8752, Recall=0.9013, Accuracy=0.8864, Loss=0.2842
Techno: F1=0.7974, Precision=0.7531, Recall=0.8472, Accuracy=0.7847, Loss=0.4661
