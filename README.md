# Fresh Banana Classification

This repo contains the code produced for the term project of CS670 - Artificial Intelligence - Fall 2023 course from Department of Computer Science of Ying Wu College of Computing at New Jersey Institute of Technology.

The project team is composed by:

* Wellington Cunha (wc44@njit.edu)
* Chandrashekhar Deginal (cd459@njit.edu)
* Dhananjay Jagdish Dubey (dd573@njit.edu)

## Project proposal

The project consisted in selecting a dataset and then a problem that could be solved with that dataset. As the dataset, we decided to work with [Fresh and Rotten Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) from Kaggle and the problem of classifying fruits and vegetables freshness.

We then reviewed 85 articles and selected 21 that were original work, peer-reviewed and the outcomes of research that added new knowledge to the area (classification of freshness using images). We learned that most of the solutions around classifying freshness of fruits deals either with small datasets (< 1000 images) and binary classification or have large datasets (> 10,000 images) but deals with multi-label classification (more than one type of fruit or vegetables with a minimum of 3 types × 2 labels, thus, minimum of 6 classes).

We understand that deep learning models can achieve good performance even for multi-label classification, but we also raised the question “what would be the performance for a model that was designed for one type of fruit or vegetable?”. We wonder if by working with one type of crop, therefore binary classification (fresh or not fresh),  the selection of the architecture and hyperparameters can yield a “specialized” classification model for that type of crop that has high accuracy.

Another potential gap that we found on those works was in relation to the splitting of sets for validation: only one of them has reserved a true validation set (a set that will be touched only once, at the end of the training and test phase).

So, the hypothesis we want to test in this project is if we can get a good accuracy (>95% on the test set) for the classification of freshness for one type of fruit or vegetable (we, somehow arbitrarily, elected banana as the type) and how well that would be generalized on a validation set.

## Codebase

We have been using Google colab to run the scripts below and the dataset, after downloaded from Kaggle, was place in Google Drive. Here are the notebooks produced during this research:

* **[01_select_data.ipynb](01_select_data.ipynb)**: this notebook contains the initial data exploration, along with the splitting of the set in two subsets:

    * **Validation**: contains a random sample of 10% of the total of images. This dataset will be only used at the end of the research, when the model is trained, to verify how the model developed generalizes.
    * **Train and Test**: contains the remaining records, that will be used to train and test the model during its development (details on how it will be split are provided in another notebook).

* **[02_data_standardization.ipynb](02_data_standardization.ipynb)**: this notebook contains the pre-processing tasks of resizing (to 512x512) and padding (with black pixels to keep aspect ratio) the train and test images. The images were then pre-processed and saved into another folder in Google Drive.
* **[03_data_augmentation.ipynb](03_data_augmentation.ipynb)**: in this notebook we explored PyTorch data augmentation technicques by using the current Test and Train subset to generate a new subset of images that are randomly rotated (up to 90 degrees) and flipped (vertically and horizontally). In that way, during the training and test phase we will have twice the number of images to train and test the model.
* **[04_model_training.ipynb](04_model_training.ipynb)**: we use this notebook to explore/experiment on building a simple Neural Network using SoftMax, including the steps required to set initial parameters and train and validate the accuracy of the model. We also start exploring Transfer Learning, all of that using [PyTorch](https://pytorch.org/).
* **[05_transfer_learning.ipynb](05_transfer_learning.ipynb)**: in this notebook we have full script with the architecture and hyperparameters that yielded the best results (ResNet18 - 100% of accuracy!).
* **[06_validating_model.ipynb](06_validating_model.ipynb)**: this notebook contains the validation using the validation dataset we have reserved at the very beginning of this experiment. Again we could achieve 100% of accuracy.

The folder [saved_model](saved_model) contains the saved model that was trained on the notebook [05_transfer_learning.ipynb](05_transfer_learning.ipynb).

