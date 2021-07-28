import os.path


def get_dirs() -> dict:
    project = os.getcwd()
    res = os.path.join(project, 'res')
    cp = os.path.join(project, 'cp')
    cp_gesture = os.path.join(cp, 'cp_gesture', 'gestureNN')

    summary = os.path.join(project, 'summary')

    testing = os.path.join(res, 'testing')
    asl_te = os.path.join(testing, 'asl')
    mnist_te = os.path.join(testing, 'mnist')
    my_data_te = os.path.join(testing, 'my_data')
    training = os.path.join(res, 'training')
    mnist_tr = os.path.join(training, 'mnist')
    asl_tr = os.path.join(training, 'asl')

    src = os.path.join(project, 'src')

    dir_dict = {'project': project, 'res': res, 'cp_gesture': cp_gesture, 'summary': summary, 'cp': cp,
                'my_data_te': my_data_te, 'asl_tr': asl_tr, 'asl_te': asl_te, 'mnist_tr': mnist_tr, 'mnist_te': mnist_te, 'src': src}

    return dir_dict


class error_msgs:
    dir_not_existing = 'Directory does not exist'
    val_split_value_not_valid = 'Val split value is not valid, must be between 0 and 1'
    not_letter = 'Input does not only contain letters'
    dataset_not_dir = 'Dataset is not a directory'
    dim_mismatch = 'Dimension of array is mismatched'
    not_a_valid_scale_factor = 'Scale factor is either to high or too low'
