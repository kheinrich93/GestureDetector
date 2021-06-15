import os.path


def get_dirs():
    project = os.getcwd()
    res = os.path.join(project, 'res')
    src = os.path.join(project, 'src')

    dir_dict = {'project': project, 'res': res, 'src': src}

    return dir_dict
