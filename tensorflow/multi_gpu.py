import tensorflow as tf

def average_gradients(tower_grads):
    """
    Average gradients over towers
    
    Args:
        list of tuple of (gradient-variable) pair for all variables
        ex) ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN)) for variable 0
        N is number of GPU towers
    Return:
        list of (gradient-variable) where gradient is averaged gradients over towers
    """
    with tf.name_scope('average_gradients'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, v in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            try:
                # Average over the 'tower' dimension
                grad = tf.concat(axis=0, values=grads)
                grad = tf.reduce_mean(grad, 0)

                # Variables are redundant because they are shared across towers.
                # => Return the first tower's pointer to the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
            except ValueError as e:
                if grads == list():
                    pass
                else:
                    print(e)
                    break

        return average_grads
