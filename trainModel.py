def cnn_model_fn(origImages_, filteredImages, imageSize):
    input_layer = tf.reshape(origImages_, [-1, imageSize[0], imageSize[1], 1])
    filtered_images = tf.reshape(filteredImages, [-1, imageSize[0], imageSize[1], 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[3, 3],
        padding="same",
        strides=(1, 1),
        activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        strides=(2, 2),
        activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=(2, 2),
        activation=tf.nn.relu)

    dconv1 = tf.layers.conv2d_transpose(
        inputs=conv3,
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    dconv1 = tf.concat((dconv1, conv2), axis=3)

    dconv2 = tf.layers.conv2d_transpose(
        inputs=dconv1,
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu)

    dconv2 = tf.concat((dconv2, conv1), axis=3)

    dconv3 = tf.layers.conv2d_transpose(
        inputs=dconv2,
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    dconv3 = tf.concat((dconv3, input_layer), axis=3)

    output_ce = tf.layers.conv2d(
        inputs=dconv3,
        filters=2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    output_mse = tf.layers.conv2d(
        inputs=dconv3,
        filters=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same")

    loss1 = tf.reduce_mean(tf.norm(tf.subtract(output_mse, filtered_images)))
    output_ce = tf.nn.softmax(output_ce, axis=3)

    loss2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(filteredImages, output_ce)

    # return (loss(filteredImages, output), output)
    return (loss1 + loss2, output_ce)