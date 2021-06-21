import os.path


def get_dirs():
    project = os.getcwd()
    res = os.path.join(project, 'res')
    cp = os.path.join(project, 'cp')
    cp_gesture = os.path.join(cp, 'cp_gesture')

    summary = os.path.join(project, 'summary')

    testing = os.path.join(res, 'testing')
    asl_te = os.path.join(testing, 'asl')
    my_data_te = os.path.join(testing, 'my_data')
    training = os.path.join(res, 'training')
    asl_tr = os.path.join(training, 'asl')
    my_data_tr = os.path.join(training, 'my_data')

    src = os.path.join(project, 'src')

    dir_dict = {'project': project, 'res': res, 'cp_gesture': cp_gesture, 'summary': summary,
                'my_data_te': my_data_te, 'asl_tr': asl_tr, 'asl_te': asl_te, 'src': src}

    return dir_dict
