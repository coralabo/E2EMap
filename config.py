import argparse

parser = argparse.ArgumentParser(description='Configuration file')

net_arg = parser.add_argument_group('Network')
net_arg.add_argument('--batch_size', type=int, default=64, help='batch size')
net_arg.add_argument('--pea_width', type=int, default=4, help='width of pea')
net_arg.add_argument('--src_file_path', type=str, default="data/input.txt",
                     help='file path of the source graph')
net_arg.add_argument('--gcn_dims', type=int, default=64, help='Dimensions in hidden layers')
net_arg.add_argument('--head_nums', type=int, default=4, help='Nums of head')
net_arg.add_argument('--actor_lr', type=float, default=1e-4, help='learning rate in network')
net_arg.add_argument('--max_iteration', type=int, default=1000, help='max iteration')
net_arg.add_argument('--ckpt_dir', type=str, default='tmp', help="path to save models")
net_arg.add_argument('--temperature', type=float, default=10, help="temperature of softmax")
net_arg.add_argument('--beta', type=float, default=0.2, help="parameter of extra reward")
net_arg.add_argument('--load_model', type=bool, default=False, help="whether use previous model")
net_arg.add_argument('--layer_nums', type=int, default=14, help="nums of hidden layers")
net_arg.add_argument('--c', type=float, default=5, help="constant for tanh layer")
net_arg.add_argument('--mii', type=int, default=2, help="the minimum ii")

net_arg.add_argument('--max_LRF', type=float, default=2, help="Maximum number of LRFs in a PEA")
net_arg.add_argument('--max_GRF', type=float, default=2, help="Maximum number of GRFs in a layer")
net_arg.add_argument('--max_memory', type=float, default=0, help="How many memory resources are there at most in a layer")
net_arg.add_argument('--memory_mode', type=bool, default=False, help="Set the opening or closing of memory")
net_arg.add_argument('--reward_mode', type=int, default=1, help="1 mean mesh,2 mean mesh+torus,3 mean mesh+diagonal,4 mean mesh+1-hop,5 mean mesh+torus+diagonal+1-hop,6 mean mesh+torus+1-hop")


def get_config():
    config, _ = parser.parse_known_args()
    return config
