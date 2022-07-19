
# Potato Leaf Disease Classification

Mini Project 6th Semester

Contributors:
- Rakshit Acharya
- Adhitya Narayan P A 

Base Paper Link : https://doi.org/10.1016/j.pmpp.2021.101781




## Proposed Work
The objective of this project is to present a system to classify the leaves into the three classes i.e. two types of diseases and healthy categories based on the leaf condition by utilising deep learning techniques. Transfer Learning methods are used for classification and the models are compared based on various evaluation metrics such as precision, recall and accuracy. The various models used for classification are VGG16, VGG19, RESNET50, and MOBILENET. An interactive GUI was built for predicting if the newly uploaded image of a potato leaf is early blight, late blight, or healthy.

### Methodology
The proposed methodology for classifying the potato leaf diseases is carried out in five phases. The work flow of the steps involved is shown in Figure 3.1.1.
- Dataset Collection
- Data Preprocessing
    - RandomOverSampler
    - ImageDataGenerator
- Implementing the models
- Choosing the best performing model
- Interactive web app for prediction.

### Dataset Collection:
The dataset for the potato leaf diseases was obtained from the PlantVillage dataset, which was published by David. P. Hughes, Marcel Salathe. The datasets include over 50,000 images of 14 crops such as potatoes, grapes, tomato, apples etc.\
From the 14 classes of data we have focused only on the 3 Potato leaf classes.

Total number of images = 2152
Number of images in Early_blight = 1000\
Number of images in Late_blight = 1000\
Number of images in healthy = 152

### Data Preprocessing:
The number of images in the healthy category is 152 and the other categories have 1000 each. The imbalance in the data is of the ratio 1 : 6.5. Imbalanced classifications pose a challenge for predictive modelling as most of the deep learning algorithms will have poor predictive accuracy as there isn’t enough data for the model to learn the features of the minority class thus the images from the minority class will be classified wrongly.
#### RandomOverSampler: (Novelty)
To solve the above problem, RandomOverSampler was used which randomly chooses samples from the minority class and duplicates them with replacement in the dataset. This automatically balances the minority classes with the majority class/classes. The major demerit of this method is that it may cause overfitting of data in few models as the same samples of data are being duplicated and used for training purposes.

#### ImageDataGenerator:
It is an image augmentation technique that is used when there is not enough data to train the model. This is a great way to expand the size of our dataset. The transformation techniques used were horizontal_flip, rotation_range, zoom_range, width_shift_range, height_shift_range. Using these techniques, new transformed images from the original dataset will be used to train the model. This also helps in the model learning various images in different angles and perspectives and thus reduces the chances of the model overfitting the data.

### Model Implementation
A convolutional neural network (CNN) is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data. In this study, several deep learning models, namely VGG16, VGG19, MobileNet, and ResNet50 were trained to classify. The results of these models were evaluated on the basis of a few metrics such as accuracy, precision and recall. Transfer learning from the imagenet dataset was used where the model is pre-trained and the weights are being used for classification of images on a different problem. All four models were trained on the training dataset and ResNet50 gave the best results with around 99.77% validation accuracy.
The models were run for 20 epochs; batch size = 64,optimizer used - RMSprop, test size = 14%, validation split = 33% i.e. number of images in training dataset 2585, testing dataset = 421, validation = 853 images.
#### Choosing the best performing model
Among all the four given models, ResNet50 outperforms the other models. The model after training for 20 epochs gave ~99.77% validation accuracy and 99%, 98%, 99% precision on the 3 classes respectively. During the training of the model, first the raw conventional base of the model was imported with all the layers (convolutional and pooling layers), and the weights were initialised using ImageNet. On top of these layers a classifier was added. The output from the conventional base is the input for the last five classifier layers, that consists of a flatten layer and three dense layers and finally the output layer will be used to predict the image as any of the three classes.
### Interactive WebApp for prediction( GUI ):
An interactive web application was developed which uses the ResNet50 model trained on the given dataset for classifying newly uploaded images.
The web-app was developed using the streamlit package in python. Streamlit is an open source app framework which is used to build webapps for machine learning and data science related projects in a short period of time.
## Conclusion
In this study, a three-class classification problem is exhibited to detect infected potato leaves as early blight or late blight or healthy, in the earlier stages of the infection so that it can be treated as soon as possible. Considerable data augmentation using ImageDataGenerator and RandomOverSample is done on the data due to the imbalance in the dataset, and the data is then used to train various transfer learning CNN models, namely VGG16, VGG19, Mobilenet and ResNet50, and the results obtained are evaluated and assessed.
The results show that ResNet50 performed the best among these four, achieving an validation accuracy of ~99.77% and further has been optimised to provide better results. Further, with these models, a WebApp GUI has been built to facilitate the use of these models by farmers and other relevant communities. We beleive that this work will be a notable contribution to the sector of agriculture. We hope that this work proves useful to the research communities to get familiar with the AI-based detection of potato leaves’ diseases. The model developed not only works on the benchmark dataset, but can also work on real-time images with a decent accuracy.
In the future, the same work can be extended to include all classes of plants like tomato, bell pepper, etc., and the accuracy for classifying real time images can be improved.