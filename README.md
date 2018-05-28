# Udacity Self-Driving car Nano-degree Term 3
# Project 12: Semantic Segmentation-Project


## 1. Introduction
Fully convolutional network (FCN)

The neural network looks as follows:

![Image of fully convilutional network](./images/VGG.png)


## 2. The Program
Apart from the `run()`-function which is the entry point of the program, there are four functions in `main.py`: `load_vgg()`, `layers()`, `optimize()`, and `train_nn()`.

`load_vgg()`: Loads a pretrained VGG-model and returns handles for the internal layers, the input and output layers, and for the dropout parameter. These handles will be later needed to define the FCN, and feeding when training.

`layers()`: This function defines the graph for the fully

`optimize()`: Chooses an optimization operation. In our case, we chose an Adam-optimizer (source).

`train_nn()`: This function trains the Fully-convolutional neural network. However, before doing that, it will initialize all variables of the network first.

`gen_batch_function()`: This function returns a generating function which creates the mini-batches for the training loop. The generating function (`get_batches_fn()`) itself returns two lists of input images and label images of prescribed length each.

The main task of the `run()`-function is to open (and eventually close) a Tensorflow-session `sess` and call all of the above functions one by one to define and train the FCN.

After the FCN has been trained, the helper function `helper.save_inference_samples()` is being called which.


## 2. How to run the Project
The project can be run by launching `python main.py`.
The main script launches and loads a pretrained VGG-model.
It proceeds to train the new convoluted neural network using the .


## 3. Results
The model was trained on a workstation with a 6th generation Intel i5 CPU, 16GB RAM, and a Nvidia GTX1060 GPU with 6GB VRAM. Due to memory constraints the batch size, parametrized by `BATCH_SIZE`, was restricted to 8.

### Attempt to calculate IoU:
