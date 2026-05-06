# Computer Vision Climbing Hold Classifier

## Report for Project 03
[Report](https://docs.google.com/document/d/1WIg514gla4wTKN58F-gJ0mYcQYed7jNy8y2Kp3ostzQ/edit?usp=sharing)

### How to Run:
First, install all dependencies in `requirements.txt`

To run individual stages of the pipeline, simply run their dedicated python files.
Note that these must be run in order to have the correct images saved
- `python preprocessing.py`
- `python segmentation_anything.py`
- `python feature_extraction_sift.py`

To run the entire pipeline through the SVM, simply run `python svm_classification.py`
- This does not require images to be generated at each step via the above python files

A virtual environment is highly recommended to run this project

## Project 4
For the fourth part of the project, I trained six different models with these pipelines:
1. Normalization --> SAM Segmentation --> SIFT-Based Feature Extraction + Regionprops Local Features --> SVM Linear Classification (implemented in project 3)
2. Normalization --> SAM Segmentation --> SIFT-Based Feature Extraction + Regionprops Local Features --> SVM RBF Classification    (implemented in project 3)
3. Normalization --> SAM Segmentation --> RESNET CNN Feature Extraction --> SVM Linear Classification
4. Normalization --> SAM Segmentation --> RESNET CNN Feature Extraction --> SVM RBF Classification
5. Normalization --> SAM Segmentation --> RESNET CNN Classification (results taken from results of training RESNET in models 3 and 4)
6. Normalization --> RESNET CNN Classification on Cropped & Normalized Image
7. Normalization --> SAM Segmentation --> Custom CNN Feature Extraction --> SVM RBF Classification
8. Normalization --> Custom CNN Classification (results taken from results of training classifier in model 7)
