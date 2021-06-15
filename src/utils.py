import os.path


def get_dirs():
    project = os.getcwd()
    res = os.path.join(project, 'res')

    testing = os.path.join(res, 'testing')
    asl_te = os.path.join(testing, 'asl')
    my_data_te = os.path.join(testing, 'my_data')
    training = os.path.join(res, 'training')
    asl_tr = os.path.join(training, 'asl')
    my_data_tr = os.path.join(training, 'my_data')

    src = os.path.join(project, 'src')

    dir_dict = {'project': project, 'res': res,
                'my_data_te': my_data_te, 'asl_tr': asl_tr, 'src': src}

    return dir_dict
