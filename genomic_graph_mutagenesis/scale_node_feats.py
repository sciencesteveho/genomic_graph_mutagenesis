#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Code to scale node_feats
"""

import argparse
import joblib
import pickle

import numpy as np

from utils import dir_check_make


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tissue', type=str, default='mammary',
                        help='tissue_type')
    args = parser.parse_args()

    root_dir = '/ocean/projects/bio210019p/stevesho/data/preprocess'
    scale_dir=f'{root_dir}/data_scaler'
    graph_dir=f'/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/{args.tissue}'
    out_dir=f'{graph_dir}/scaled'
    dir_check_make(out_dir)

    scalers = {i: joblib.load(f'{scale_dir}/feat_{i}_scaler.pt') for i in range(0, 36)}

    with open(f'{graph_dir}/{args.tissue}_full_graph.pkl', 'rb') as f:
        g = pickle.load(f)
    node_feat = g['node_feat']
    for i in range(0, 36):
        node_feat[:,i] = scalers[i].transform(node_feat[:,i].reshape(-1,1)).reshape(1, -1)[0]
    g['node_feat'] = node_feat 
    with open(f'{out_dir}/graph.pkl', 'wb') as output:
        pickle.dump(g, output)