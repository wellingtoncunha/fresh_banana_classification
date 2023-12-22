# Fresh Banana Classification

## Project proposal

The project consisted in selecting a dataset and then a problem that could be solved with that dataset. As the dataset, we decided to work with [Fresh and Rotten Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) from Kaggle and the problem of classifying fruits and vegetables freshness.

We then reviewed 85 articles and selected 21 that were original work, peer-reviewed and the outcomes of research that added new knowledge to the area (classification of freshness using images). We learned that most of the solutions around classifying freshness of fruits deals either with small datasets (< 1000 images) and binary classification or have large datasets (> 10,000 images) but deals with multi-label classification (more than one type of fruit or vegetables with a minimum of 3 types × 2 labels, thus, minimum of 6 classes).

We understand that deep learning models can achieve good performance even for multi-label classification, but we also raised the question “what would be the performance for a model that was designed for one type of fruit or vegetable?”. We wonder if by working with one type of crop, therefore binary classification (fresh or not fresh),  the selection of the architecture and hyperparameters can yield a “specialized” classification model for that type of crop that has high accuracy.

Another potential gap that we found on those works was in relation to the splitting of sets for validation: only one of them has reserved a true validation set (a set that will be touched only once, at the end of the training and test phase).

So, the hypothesis we want to test in this project is if we can get a good accuracy (>95% on the test set) for the classification of freshness for one type of fruit or vegetable and how well that would be generalized on a validation set.

## Codebase

We have been using Google colab to run the scripts below and the dataset, after downloaded from Kaggle, was placed in Google Drive. Here are the notebooks produced during this research:

* **[01_data_exploration.ipynb](01_data_exploration.ipynb)**: this notebook contains the initial data exploration, along with the splitting of the set in two subsets:

    * **Validation**: contains a random sample of 10% of the total of images. This dataset will be only used at the end of the research, when the model is trained, to verify how the model developed generalizes.
    * **Train and Test**: contains the remaining records, that will be used to train and test the model during its development (details on how it will be split are provided in another notebook).

* **[02_data_standardization.ipynb](02_data_standardization.ipynb)**: this notebook contains the pre-training process of padding (with black pixels to keep aspect ratio) and resizing (to 512x512) the train and test images. The images were then pre-processed and saved into another folder in Google Drive using a pattern for their name and in png format.
* **[03_data_augmentation.ipynb](03_data_augmentation.ipynb)**: in this notebook we explored data augmentation, first using OpenCV to rotate (up to i0 degrees) and flip (vertically and horizontally) and next using PyTorch to do that. The whole idead is to double the number of images for development and training/testing phase by using Data Augmentation.
* **[04_augmented_images_statistics.ipynb](04_augmented_images_statistics.ipynb)**: in this notebook we get some statistics (mean and std) of images after transformation and augmentation. We don't capture min and max values of each channel because we learned, during exploration, that EVERY picture will contain at least one white (0, 0, 0) and one black (255, 255, 255) pixels, so, the min value of each RGB channel will always be 0, while the max value will always be 255.
* **[05_simple_nn_model_training.ipynb](05_simple_nn_model_training.ipynb)**: we use this notebool to explore/experiment on building a simple Neural Network using SoftMax, including the steps required to set initial parameters and train and validate the accuracy of the model. 
* **[06_transfer_learning_model_training.ipynb](06_transfer_learning_model_training.ipynb)**: in this notebook we have a full script with the architecture and hyperparameters that yielded the best results (ResNet18 - 100% of accuracy!).
* **[07_validating_model.ipynb](07_validating_model.ipynb)**: this notebook contains the validation using the validation dataset we have reserved at the very beginning of this experiment. Again we could achieve 100% of accuracy.
* **[08_checking_with_real_world_images.ipynb](08_checking_with_real_world_images.ipynb)**: we took some photos of bananas using a mobile device, place them on [photos](photos) folder and then used the model to predict. All the codes are available in this notebook.

The folder [saved_model](saved_model) contains the saved model that was trained on the notebook
