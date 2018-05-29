import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


"""
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_X, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={input_image: batch_X, correct_label: batch_y})
        total_accuracy += (accuracy*len(batch_X))

    return total_accuracy / num_examples
"""




def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #tf.train.Saver().load(sess, [vgg_tag], vgg_tag)
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, strides=(1,1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, strides=(1,1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=(1,1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    # Upsample 2x
    output = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, kernel_size=4, strides=(2,2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    layer4_conv_1x1 = tf.nn.max_pool(layer4_conv_1x1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    output_deconv_1 = tf.add(output, layer4_conv_1x1)

    # Upsample 2x
    output = tf.layers.conv2d_transpose(output_deconv_1, num_classes, kernel_size=4, strides=(8,8), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    #layer3_conv_1x1 = tf.nn.max_pool(layer3_conv_1x1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    output_deconv_2 = tf.add(output, layer3_conv_1x1)

    # Upsample 8x
    output = tf.layers.conv2d_transpose(output_deconv_2, num_classes, kernel_size=16, strides=(8,8), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    return output


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # Flatten images
    logits = tf.reshape(nn_last_layer, (-1, num_classes)) # Does the input need to be rescaled?
    correct_label_flattened = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    loss_operation = tf.reduce_mean(cross_entropy_loss)

    # Add regularization losses here:
    #regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    #loss_operation += regularization_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_operation)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    # Creating a new session is not necessary here
    sess.run(tf.global_variables_initializer())

    #saver = tf.train.Saver()
    for epoch in range(epochs):
        counter=0
        for image_batch, label_batch in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={ input_image: image_batch, correct_label: label_batch, keep_prob: 0.8 })
            print("Batch No. {}, Epoch: {}/{}".format(counter, epoch+1, epochs))
            print("Mean cross entropy loss: {}".format(np.mean(loss)))
            counter += 1
    
    #tf.train.export_meta_graph('./meta/model.meta')
    #tf.train.write_graph(sess.graph_def, './checkpoints/', 'model.pb', False)
    #saver.save(sess, './model')
    #tf.saved_model.simple_save(sess, './model1', inputs={ "input_image": input_image}, outputs={ "correct_label": correct_label})





tests.test_train_nn(train_nn)


def run():
    NUM_CLASSES = 2 # class 1: 'Road'; class 2: 'Background'
    IMAGE_SHAPE = (160, 576)
    DATA_DIR = './data'
    RUNS_DIR = './runs'
    tests.test_for_kitti_dataset(DATA_DIR)

    VGG_PATH = os.path.join(DATA_DIR, 'vgg')

    EPOCHS = 10 # 10
    BATCH_SIZE = 8  # max. 8 on a GTX1060, 6GB VRAM
    LEARNING_RATE = 0.001

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create generating function for batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Loads data from the pretrained vgg-model:
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, VGG_PATH)
        
        # Defines the tensor:
        decoder_output = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)
        correct_label = tf.placeholder(dtype=tf.bool, shape=(None, 160, 576, 2))
        
        logits, train_op, cross_entropy_loss = optimize(decoder_output, correct_label, LEARNING_RATE, NUM_CLASSES)

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, LEARNING_RATE)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video




if __name__ == '__main__':
    run()





"""
with tf.Session(graph=g) as sess:
    tf.train.Saver().restore(sess, './model')   
    testing_accuracy = evaluate(X_test_gray, y_test)
    print("The testing accuracy is: {}".format(testing_accuracy))
    training_accuracy = evaluate(X_train_gray, y_train)
    print("The accuracy on the training set is: {}".format(training_accuracy))
    validation_accuracy = evaluate(X_valid_gray, y_valid)
    print("The accuracy on the validation set is: {}".format(validation_accuracy))
"""
"""
with tf.Session(graph = g) as sess:
    tf.train.Saver().restore(sess,'./model')
    top_k = tf.nn.top_k(sess.run(fc3, feed_dict={X: img_gray}), k=5, sorted=True)
    print(sess.run(top_k))
"""