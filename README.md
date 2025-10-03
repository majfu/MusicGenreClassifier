My first bigger machine learning project, I am very proud of it (˶ᵔ ᵕ ᵔ˶)

I trained models for music genre classification task. The first approach was a single model for multilabel prediction, but the performance was mediocre. The second approach was to train a binary classifier for each genre, and here the performance improved. Combining the models predictions seems to yield even better results.

There is also a frontend written in React (my first time working with React) and a Falsk backend to process the audio files.


Results of model evaluation on test dataset

Multilabel:
Test Loss: 0.2805
Macro F1 Score: 0.6073

Multilabel per-genre:
| Genre       | F1    | Precision | Recall | Accuracy |
|-------------|-------|-----------|--------|----------|
| Classical   | 0.8592 | 0.7625   | 0.9839 | 0.9682 |
| Dance       | 0.3514 | 0.5417   | 0.2600 | 0.9237 |
| Electronic  | 0.6087 | 0.7179   | 0.5283 | 0.8855 |
| Experimental| 0.6010 | 0.6237   | 0.5800 | 0.8776 |
| Hip-Hop     | 0.7059 | 0.8571   | 0.6000 | 0.9205 |
| Metal       | 0.7059 | 0.6923   | 0.7200 | 0.9523 |
| Old-Time    | 0.9903 | 0.9808   | 1.0000 | 0.9984 |
| Pop         | 0.3648 | 0.4915   | 0.2900 | 0.8394 |
| Punk        | 0.3256 | 0.3889   | 0.2800 | 0.9078 |
| Rock        | 0.7945 | 0.7982   | 0.7909 | 0.9285 |
| Techno      | 0.3733 | 0.5600   | 0.2800 | 0.9253 |


Binary model for each genre:
| Genre       | F1    | Precision | Recall | Accuracy | Loss   |
|-------------|-------|-----------|--------|----------|--------|
| Classical   | 0.9381 | 0.9138   | 0.9636 | 0.9364 | 0.2948 |
| Dance       | 0.7708 | 0.7255   | 0.8222 | 0.7556 | 0.5850 |
| Electronic  | 0.8732 | 0.8732   | 0.8732 | 0.8732 | 0.3014 |
| Experimental| 0.7892 | 0.7816   | 0.7970 | 0.7871 | 0.4308 |
| Hip-Hop     | 0.9119 | 0.9312   | 0.8934 | 0.9137 | 0.2474 |
| Metal       | 0.7925 | 0.7778   | 0.8077 | 0.7885 | 0.3995 |
| Old-Time    | 0.9778 | 0.9778   | 0.9778 | 0.9778 | 0.0321 |
| Pop         | 0.7081 | 0.7184   | 0.6981 | 0.7123 | 0.5730 |
| Punk        | 0.8403 | 0.8018   | 0.8826 | 0.8322 | 0.3996 |
| Rock        | 0.8880 | 0.8752   | 0.9013 | 0.8864 | 0.2842 |
| Techno      | 0.7974 | 0.7531   | 0.8472 | 0.7847 | 0.4661 |
