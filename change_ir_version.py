import sys
import numpy
import onnx

import onnxruntime as rt

if __name__ == '__main__':
    # Checks
    model_onnx = onnx.load(sys.argv[1])  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    model_onnx.ir_version = 8

    onnx.save(model_onnx, sys.argv[1])
