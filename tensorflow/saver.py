

def optimistic_restore(sess, ckpt_path, restore_vars=None, name=None):
    """
    Restore all variables in checkpoint (ckpt_path)
    """
    # Reference: https://github.com/tensorflow/tensorflow/issues/312

    # Load checkpoint reader
    reader = tf.train.NewCheckpointReader(ckpt_path)

    # All variable saved in checkpoint
    # Dict: name => shape
    # {'global_step': [],
    # 'resnet_v1_101/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta': [64], .. }
    saved_shapes = reader.get_variable_to_shape_map()

    # List of all names of global variables in current graph
    # Sort because variables in checkpoints are sorted by their names already.
    # [('global_step:0', 'global_step'),
    # ('resnet_v1_101/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0', .. ]
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])

    # Dict: name => variable
    # Key: 'Decoder/LSTM_initializer/Layer_0/fully_connected/biases'
    # Value: <tf.Variable 'Decoder/LSTM_initializer/Layer_0/fully_connected/biases:0' shape=(512,) dtype=float32_ref>
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

    # List all global variables to restore if they are in checkpoint
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)

    # Restore variables
    saver = tf.train.Saver(restore_vars, name=name)
    saver.restore(sess, ckpt_path)

def get_uninit_vars(sess, variables=None):
    """
    Return list of uninitialized variables
    """
    if variables is None:
        variables = tf.global_variables()
    init_flag = sess.run(
        tf.stack([tf.is_variable_initialized(v) for v in variables]))
    return [v for v, f in zip(variables, init_flag) if not f]
