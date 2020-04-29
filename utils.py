# -*- coding: utf-8 -*-

import os
import pathlib

def read_label(label_path):

    '''
    Read label file and return the list of absolute paths
    '''

    with open(label_path, 'r') as f:
        files = f.readlines()
        
    dirname = os.path.dirname(label_path)
    abs_paths = []
    for fp in files:
        fp = fp[:-1] # remove '\n'
        rel_path = pathlib.Path(os.path.join(dirname, fp))
        abs_path = rel_path.resolve()
        abs_paths.append(str(abs_path))

    return abs_paths

if __name__=='__main__':
    print(read_label('data/label/train_denoise'))
        
