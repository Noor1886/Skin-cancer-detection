# Skin-cancer-detection
## Detection_of_Malignant_Skin_Cancer_using_a_CNN
### Objective:

To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
Summary of the Skin Cancer Data:

    Total number of images: 2357
    Images consist of Bening and Malignant Oncological Diseases.
    Data was gathered by ISIC (International Skin Imaging Collaboration).
    Images were sorted according to the classification taken with ISIC.
    All the subsets were divided into the same number of images, with an excpetion for melanomas whose images are slightly dominant.

### Diseases in the Dataset

    Actinic keratosis
    Basal cell carcinoma
    Dermatofibroma
    Melanoma
    Nevus
    Pigmented benign keratosis
    Seborrheic keratosis
    Squamous cell carcinoma
    Vascular lesion

### CNN Model Architecture

    The CNN is a Sequential Model type which allows us to construct our Neural Network Architecture layer by layer
    The Architecture will be a combination of 2D convolution layers along with Max pooling in this architecture.
    We will be using activation functions Softmax and RelU.
    Why am i using ReLU?, Because it is simple, effective and addresses the problem of vanishing gradients.
    Why am i using Softmax?, Because it makes interpreting the outputs of the Neural Network easier in the case of Multi-Class Classification by normalizing the outputs to probablities.

Architecture -> Rescaling + Convolution + Pooling + Dropout + Flatten + Dense

    Rescaling: rescaling the input data ensures that all features are on a similar scale.
        This is particularly important when working with different types of features, such as pixel intensities and spatial coordinates, as having features with disparate scales can lead to biased or inefficient learning.
        a rescaling layer also helps to mitigate the influence of outliers in the data. Outliers, which are data points that significantly deviate from the majority of the data, can adversely affect the model accuracy.
    Convolution: It is specifically designed to process and analyze spatial structures in data such as images.
        A convolutional layer applies a set of learnable filters to the input data, convolving them with the input pixels to extract and learn meaningful features.
        These filters act as feature detectors, capturing different characteristics like edges, textures, or patterns present in the input.
        By repeatedly applying these filters, a convolutional layer hierarchically learns and abstracts more complex features at different scales.
    Pooling: Also known as a subsampling layer, it's primary purpose is to reduce the spatial dimensions (width and height) of the input volume, while retaining the most important information.
        They help to reduce the computational complexity of the network by decreasing the number of parameters and operations. This allows the CNN to process larger input volumes more efficiently.
        Pooling layers introduce a form of translation invariance in the network by aggregating neighboring features and extracting their most representative elements.
    Dropout: This layer is used for regularization purposes. It helps reduce overfitting, which occurs when the model becomes too specialized to the training data.
        A dropout layer randomly selects a fraction of the neurons (typically 20-50%) in a neural network and temporarily "drops out" or turns off these neurons during training. This means that these neurons do not contribute to the forward and backward pass of the network, effectively reducing the complexity and capacity of the model.
        By dropping out neurons, the model becomes less reliant on a single set of features and learns to create redundant representations. This helps in creating more robust and generalized features, thereby improving the model's ability to generalize and perform well on unseen data.
    Flatten: It is primarily used to transform multidimensional data into a one-dimensional array.
        In neural networks, data is typically represented in the form of tensors, which are multi-dimensional arrays. However, many algorithms and functions in machine learning and deep learning are designed to work with one-dimensional input.
        This is where the flatten layer comes into play. The flatten layer takes the multi-dimensional input and reshapes it into a one-dimensional array. This is done by sequentially stacking all the elements of the input array one after another. The resulting flattened array can then be easily fed into the subsequent layers of the network.
    Dense: It is used to connect every neuron from the previous layer to the neurons in the current layer.
        The purpose of a dense layer is to allow the network to learn complex patterns and relationships in the data.
        By connecting all the neurons in the previous layer to the current layer, the dense layer can capture interactions between all the features present in the input data.
        This enables the network to make non-linear transformations and extract higher-level representations.

### Model Compilation

    We are using the Adam Optimizer for adjusting the parameters of a neural network in real-time to improve its accuracy and speed.

### Why do we compile a model?

    We compile a CNN model in order to specify various important settings and configurations for training the model.
    First, compiling a CNN model allows us to choose an appropriate optimizer, which is responsible for updating the model's weights based on the computed gradients. Different optimizers, such as SGD, Adam, or RMSprop, have different properties and can affect the model's training speed and final performance.
    Second, by compiling the model, we can specify the desired loss function. The loss function measures the difference between the predicted outputs and the true outputs, and it plays a crucial role in guiding the learning process.

### Adam Optimizer: It is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. Adam optimization algorithm combines the concepts of adaptive learning rates and momentum to adjust the update values for the model's parameters. Adam stand for Adaptive Moment Estimation, which means that it adapts the learning rate of each parameter based on its historical gradients and momentum

    The main advantage of using the Adam optimizer is its ability to handle both sparse and noisy gradients effectively. It adapts the learning rate for each parameter separately, allowing for efficient and accurate updates. This helps in avoiding getting stuck in local optima and speeds up the training process.
    Another reason why we use the Adam optimizer is its ability to handle large-scale problems. It dynamically adjusts the learning rates based on the estimated second moment of the gradients, making it suitable for models with a large number of parameters.

### Stochastic Gradient Descent: Stochastic gradient descent (SGD) is an optimization algorithm commonly used in machine learning and deep learning. It is an extension of gradient descent, but instead of updating the model parameters after evaluating the entire dataset, SGD updates the parameters after each individual sample or a small batch of samples.

    SGD offers several advantages over traditional gradient descent. Firstly, it is computationally efficient since it only requires a small batch of samples to compute the gradients and update the parameters, as opposed to processing the entire dataset. This is particularly useful when working with large datasets.
    Furthermore, SGD enhances model generalization by introducing randomness during parameter updates. This randomness helps the model escape from local minima and find better solutions in the local optimization landscape. This randomization also adds a form of regularization to the model, preventing overfitting by introducing noise in the parameter updates.

### Categorical_crossentropy: Categorical crossentropy is a commonly used loss function, specifically in classification tasks that involve multi-class problems. It is used to measure the dissimilarity between the predicted probability distribution generated by a model and the true probability distribution.

    In multi-class classification, the output of the model is usually a set of probabilities assigned to each class. The true labels are represented as one-hot encoded vectors, where only one element has a value of 1 indicating the class label. Categorical crossentropy calculates the loss by comparing the predicted probabilities with the true labels, penalizing larger differences between them.

### Model Checkpoint: A model checkpoint is a snapshot of a machine learning model at a particular point during training. It captures the weights, biases, and other parameters of the model, allowing us to save and reuse the model for future predictions or further training.

    There are several reasons why we use model checkpoints. Firstly, they enable us to save the progress of a model during training so that we can resume training from where we left off in case of any interruptions. This is particularly useful when training deep learning models that may require days or weeks to complete.

### Early Stopping: S form of regularization to prevent overfitting and improve the generalization ability of a model. It involves monitoring the performance of the model during training and stopping the training process before it reaches the optimal number of iterations.

    The idea behind early stopping is that, initially, as the model is being trained, the performance on the training data improves. However, after a certain point, the model starts to overfit the training data, resulting in a decrease in performance on unseen data. Early stopping allows us to prevent this overfitting by stopping the training process at an earlier point when the model's performance on a validation set starts to deteriorate.

### Optimization

    We have seen the results of the previous iteration of the model, and to improve the accuracy and reduce the loss, we will we implementing some techniques to optimize the working and achieve better results.
