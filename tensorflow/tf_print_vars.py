def print_model_params(verbose=True):
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        if verbose: print("name: " + str(variable.name) + " - shape:" + str(shape))
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        if verbose: print("variable parameters: " , variable_parametes)
        total_parameters += variable_parametes
    if verbose: print("total params: ", total_parameters)
    return total_parameters
