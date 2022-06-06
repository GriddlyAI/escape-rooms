import argparse
import fnmatch
import io
import logging
import os
import re

import onnx
import torch
from onnx_tf.backend import prepare

from ppo import Agent

"""
This script converts a torch policy models to keras.

Usage:

python -m torch2keras \
--xp_path=<absolute path to folder containing torch checkpoint> \
--model_tar=<name of model.tar, without .tar part. defaults to "model"> \
--output_name=<name of output folder containing keras model>

After converting to keras, use tfjs-converter to convert to 
the resulting binary to tfjs format.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='torch2tf')

    parser.add_argument(
        '--xp_path',
        type=str,
        help='Absolute path to experiment results directory.')

    parser.add_argument(
        '--xp_prefix',
        type=str,
        help='Absolute path to experiment results directory.')

    parser.add_argument(
        '--max_seeds',
        type=int,
        default=None,
        help='Maximum number of seeds to consider when matching on xp_prefix.')

    parser.add_argument(
        '--model_tar',
        type=str,
        default='model',
        help='Name of model.tar name, without .tar extension.')

    parser.add_argument(
        '--output_dir',
        type=str,
        default='tf_models',
        help='Directory where outputs should be saved.')

    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help='Output folder name (prefix if xp_prefix provided).')

    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='Name of agent in name.txt (prefix if xp_prefix provided).')

    return parser.parse_args()


def load_griddly_model(xp_path, model_tar):
    model = None
    xp_path = os.path.expandvars(os.path.expanduser(xp_path))

    model_path = os.path.join(xp_path, f'{model_tar}.tar')
    meta_json_path = os.path.join(xp_path, f'meta.json')

    assert os.path.exists(xp_path), f'No model at {xp_path}.'
    # assert os.path.exists(meta_json_path), f'No file at {meta_json_path}.'

    # meta_json_file = open(meta_json_path)
    # xp_flags = DotDict(json.load(meta_json_file)['args'])

    model = Agent((7, 9, 51), 12)

    model_checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_checkpoint['model_state_dict'])

    return model, None


def convert2tf_and_export(model, xp_flags, export_path):
    x = torch.randn(1, 7, 9, 51, requires_grad=False)  # dummy input
    tf_model = pytorch_to_tf(model, x, [(7, 9, 51,)], change_ordering=True, verbose=True, export_path=export_path)
    return tf_model

"""
The PyTorch2Keras converter interface
"""

def pytorch_to_tf(
        model, args, input_shapes=None,
        change_ordering=False, verbose=False, name_policy=None,
        use_optimizer=False, do_constant_folding=False,
        export_path=None
):
    """
    By given PyTorch model convert layers with ONNX.

    Args:
        model: pytorch model
        args: pytorch model arguments
        input_shapes: keras input shapes (using for each InputLayer)
        change_ordering: change CHW to HWC
        verbose: verbose output
        name_policy: use short names, use random-suffix or keep original names for keras layers

    Returns:
        model: created keras model.
    """
    logger = logging.getLogger('torch2tf')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    logger.info('Converter is called.')

    if name_policy:
        logger.warning('Name policy isn\'t supported now.')

    if input_shapes:
        logger.warning('Custom shapes isn\'t supported now.')

    if input_shapes and not isinstance(input_shapes, list):
        input_shapes = [input_shapes]

    if not isinstance(args, list):
        args = [args]

    args = tuple(args)

    dummy_output = model(*args)

    if isinstance(dummy_output, torch.autograd.Variable):
        dummy_output = [dummy_output]

    input_names = ['input_{0}'.format(i) for i in range(len(args))]
    output_names = ['output_{0}'.format(i) for i in range(len(dummy_output))]

    logger.debug('Input_names:')
    logger.debug(input_names)

    logger.debug('Output_names:')
    logger.debug(output_names)

    stream = io.BytesIO()
    torch.onnx.export(model, args, stream,
                      do_constant_folding=do_constant_folding,
                      verbose=verbose,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11)

    stream.seek(0)
    onnx_model = onnx.load(stream)
    tf_rep = prepare(onnx_model)

    if export_path:
        tf_rep.export_graph(export_path)

    return tf_rep


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

if __name__ == '__main__':
    args = parse_args()
    model, xp_flags = None, None

    # Loop over all matching xpids
    if args.xp_prefix:
        base_path = os.path.expandvars(os.path.expanduser(args.xp_path))
        # Get folders matching xp_prefix in xp_path
        all_xpids = fnmatch.filter(os.listdir(base_path), f"{args.xp_prefix}*")
        filter_re = re.compile('.*_[0-9]*$')
        xpids = [x for x in all_xpids if filter_re.match(x)]
        xpids.sort()
    else:
        base_path, xp_id = os.path.split(os.path.expandvars(os.path.expanduser(args.xp_path)))
        xpids = [xp_id]

    if args.max_seeds:
        xpids = xpids[:args.max_seeds]

    for i, xpid in enumerate(xpids):
        xp_path = os.path.join(base_path, xpid)

        # Get path to model .tar
        model, xp_flags = load_griddly_model(xp_path, args.model_tar)

        # Create output dir
        output_dir = os.path.expandvars(os.path.expanduser(args.output_dir))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if args.output_name:
            output_foldername = os.path.basename(args.output_name)
            if args.xp_prefix:
                output_foldername = f'{output_foldername}_s{i}'
        else:
            output_foldername = os.path.basename(xp_id)

        output_path = os.path.join(output_dir, output_foldername)

        tf_model = convert2tf_and_export(model, xp_flags, export_path=output_path)

        # Create name file
        if args.model_name:
            with open(os.path.join(output_path, 'name.txt'), 'w') as nametxt:
                name = args.model_name
                if args.xp_prefix:
                    name = f'{name}_s{i}'
                nametxt.write(name)
