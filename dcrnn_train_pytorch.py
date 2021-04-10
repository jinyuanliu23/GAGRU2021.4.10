from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
# import tensorflow as tf
import yaml
import os
from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import GARNNSupervisor

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename','data/sensor_graph/adj_mx_bay.pkl' )
        # graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename',
        #                                                    'C:/Users/Administrator/Desktop/DCRNN_PyTorch-memoryefficiency/data/sensor_graph/adj_mx.pkl')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        # if args.use_cpu_only:
        #     tf_config = tf.ConfigProto(device_count={'GPU': 0})
        # with tf.Session(config=tf_config) as sess:
        supervisor = GARNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=os.path.join(os.getcwd(), 'data/model/dcrnn_bay.yaml'), type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')

    args = parser.parse_args()
    main(args)
