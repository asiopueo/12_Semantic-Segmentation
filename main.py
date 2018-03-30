import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


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
    
    tf.saver.loader.load(sess, [vgg_tag], vgg_tag)

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # Upsample 2x
    output = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.layers.conv2d_transpose(output, num_classes, kernel_size=4, strides=(2,2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    pool_4 = tf.nn.maxpool(vgg_layer4_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='valid')
    output_deconv_1 = tf.add(output, pool_4)

    # Upsample 2x
    output_tmp = tf.layers.conv2d(output_deconv_1, num_classes, kernel_size=1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output_tmp = tf.layers.conv2d_transpose(output_tmp, num_classes, kernel_size=4, strides=(8,8), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    pool_3 = tf.nn.maxpool(vgg_layer3_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='valid')
    output_deconv_2 = tf.add(output_tmp, pool_3)

    # Upsample 8x
    output_tmp = tf.layers.conv2d(output_deconv_2, num_classes, kernel_size=1, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.layers.conv2d_transpose(output_tmp, num_classes, kernel_size=16, strides=(8,8), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    

    # some additional stuff here...
    fcn_output = output

    return fcn_output

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

    input = correct_label
    logits = tf.reshape(input, (-1, num_classes))
    
    
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=one_hot_y)
    loss_operation = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits, labels)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate)

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
    # TODO: Implement function
    for epoch in epochs:
        for image, label in get_batches_fn(batch_size):
            # Training
            with tf.Session() as sess:
                input_image
                correct_label
                keep_prob
                sess.run(train_op, feed_dict={X: input_image, y: correct_label})

    """
    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_X, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={X: batch_X, y: batch_y})
            total_accuracy += (accuracy*len(batch_X))

        return total_accuracy / num_examples
    """

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        
        # Defines the tensor:
        decoder_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(decoder_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

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



        # TODO: Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video




if __name__ == '__main__':
    run()
