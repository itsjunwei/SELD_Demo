#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch `.h5` to ONNX Converter Script

This script loads a PyTorch model's weights from an `.h5` file, constructs the model architecture,
and exports the model to the ONNX format.
"""

import argparse
import os
import sys
import h5py
import torch
import torch.nn as nn
import torch.onnx
import onnx
import numpy as np
from models import ResNet, SELDNet

# Optional: For verification using ONNX Runtime
try:
    import onnxruntime as ort
    VERIFY_WITH_ORT = True
except ImportError:
    VERIFY_WITH_ORT = False
    print("onnxruntime not installed. Verification step will be skipped.")


#############################
#        CONFIGURATION      #
#############################

# Paths
MODEL_WEIGHTS_NAME = '030225_1553_btndsc_2fps_model'
H5_WEIGHTS_PATH = './model_weights/{}.h5'.format(MODEL_WEIGHTS_NAME)
onnx_dir = "./onnx_models"
os.makedirs(onnx_dir, exist_ok=True)
ONNX_OUTPUT_PATH = os.path.join(onnx_dir, '{}.onnx'.format(MODEL_WEIGHTS_NAME))

# Model Parameters

# Input Configuration
DUMMY_INPUT_SHAPE = (1, 7, 80, 191)  # <-- Replace with your model's expected input shape
DUMMY_INPUT_NAME = 'input'             # <-- Input tensor name
OUTPUT_NAME = 'output'                 # <-- Output tensor name
OPSET_VERSION = 13                     # <-- ONNX opset version

# Dynamic Axes (for variable batch sizes, etc.)
DYNAMIC_AXES = {
    DUMMY_INPUT_NAME: {0: 'batch_size'},
    OUTPUT_NAME: {0: 'batch_size'}
}


#############################
#    WEIGHT LOADING UTILS   #
#############################

def load_h5_state_dict(h5_path):
    """
    Loads a state dictionary from an `.h5` file.

    Args:
        h5_path (str): Path to the `.h5` file containing model weights.

    Returns:
        dict: A state dictionary compatible with PyTorch models.
    """
    state_dict = {}
    with h5py.File(h5_path, 'r') as f:
        # Inspect the structure of the .h5 file
        def printname(name):
            pass  # You can enable printing if needed for debugging
            # print(name)
        f.visit(printname)

        # Assuming weights are stored at the root level.
        # Modify the group path if your .h5 structure is different.
        for key in f.keys():
            data = f[key][()]
            state_dict[key] = torch.tensor(data)
    return state_dict

#############################
#      EXPORT FUNCTION      #
#############################

def export_to_onnx(model, dummy_input, onnx_path, input_names, output_names, dynamic_axes, opset_version=13):
    """
    Exports a PyTorch model to ONNX format.

    Args:
        model (nn.Module): The PyTorch model to export.
        dummy_input (torch.Tensor): A sample input tensor.
        onnx_path (str): Destination path for the ONNX model.
        input_names (list): List of input tensor names.
        output_names (list): List of output tensor names.
        dynamic_axes (dict): Dictionary specifying dynamic axes.
        opset_version (int, optional): ONNX opset version. Defaults to 13.
    """
    torch.onnx.export(
        model,                      # PyTorch model
        dummy_input,                # Sample input
        onnx_path,                  # Output path
        export_params=True,         # Store the trained parameters
        opset_version=opset_version, # ONNX opset version
        do_constant_folding=True,   # Optimize constants
        input_names=input_names,    # Input tensor names
        output_names=output_names,  # Output tensor names
        dynamic_axes=dynamic_axes   # Dynamic axes
    )
    print(f"Model successfully exported to {onnx_path}")

#############################
#      VERIFICATION UTILS    #
#############################

def verify_onnx_model(onnx_path, dummy_input):
    """
    Verifies the exported ONNX model using ONNX and ONNX Runtime.

    Args:
        onnx_path (str): Path to the ONNX model file.
        dummy_input (torch.Tensor): The same input used during export.
    """
    # Load the ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid.")
    except onnx.checker.ValidationError as e:
        print(f"ONNX model validation failed: {e}")
        return

    if not VERIFY_WITH_ORT:
        print("onnxruntime not available. Skipping runtime verification.")
        return

    # Initialize ONNX Runtime session
    session = ort.InferenceSession(onnx_path)

    # Prepare input
    input_name = session.get_inputs()[0].name
    input_data = dummy_input.numpy()

    # Run inference
    outputs = session.run(None, {input_name: input_data})
    print("ONNX Runtime inference successful. Output shape:", outputs[0].shape)

#############################
#          MAIN              #
#############################

def main(args):
    """
    Main function to orchestrate the conversion process.
    """
    # Configuration Overrides (if any)
    h5_weights_path = args.h5_weights if args.h5_weights else H5_WEIGHTS_PATH
    onnx_output_path = args.output if args.output else ONNX_OUTPUT_PATH
    input_shape = args.input_shape if args.input_shape else DUMMY_INPUT_SHAPE
    opset_version = args.opset if args.opset else OPSET_VERSION

    # Check if .h5 file exists
    if not os.path.isfile(h5_weights_path):
        print(f"Error: .h5 weights file not found at {h5_weights_path}")
        sys.exit(1)

    # Instantiate the model
    model = ResNet(in_feat_shape=(7,80,191),
                   out_feat_shape=(2,6),
                   btn_dsc=True,
                   use_dsc=False,
                   lightweight=False,
                   fps=2)
    print("Model architecture instantiated.")

    # Load weights from .h5
    print(f"Loading weights from {h5_weights_path}...")
    try:
        model.load_state_dict(torch.load(h5_weights_path, map_location='cpu'))
        print("Weights successfully loaded into the model.")
    except RuntimeError as e:
        print(f"Error loading state_dict into the model: {e}")
        sys.exit(1)

    # Set model to evaluation mode
    model.eval()
    print("Model set to evaluation mode.")

    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    print(f"Dummy input created with shape: {dummy_input.shape}")

    # Export to ONNX
    export_to_onnx(
        model=model,
        dummy_input=dummy_input,
        onnx_path=onnx_output_path,
        input_names=[DUMMY_INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=DYNAMIC_AXES,
        opset_version=opset_version
    )

    # Optional: Verify the exported ONNX model
    if args.verify:
        if VERIFY_WITH_ORT:
            verify_onnx_model(onnx_output_path, dummy_input)
        else:
            print("onnxruntime not installed. Install it to enable verification.")
    
    print("Conversion process completed successfully.")

#############################
#        ARGPARSE            #
#############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch `.h5` model to ONNX format.")
    parser.add_argument('--h5_weights', type=str, default=None, help='Path to the `.h5` weights file.')
    parser.add_argument('--output', type=str, default=None, help='Output path for the ONNX model.')
    parser.add_argument('--input_shape', type=int, nargs='+', default=None, help='Input shape as space-separated integers, e.g., --input_shape 1 3 224 224')
    parser.add_argument('--opset', type=int, default=None, help='ONNX opset version to use.')
    parser.add_argument('--verify', action='store_true', help='Verify the ONNX model after export using ONNX Runtime.')

    args = parser.parse_args()

    main(args)
