import os.path
import numpy as np
import csv


def get_dirs():
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


def get_imgdata_from_csv(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file, delimiter=',')
        imgs = []
        labels = []

        next(reader, None)

        for row in reader:
            label = row[0]
            data = row[1:]
            img = np.array(data).reshape((28, 28))

            imgs.append(img)
            labels.append(label)

        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)
    return images, labels
