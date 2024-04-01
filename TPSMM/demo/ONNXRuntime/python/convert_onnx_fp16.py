 # You can clone the source code of onnxruntime to run this script as the following:
 #    git clone https://github.com/microsoft/onnxruntime
 #    cd onnxruntime/onnxruntime/python/tools/transformers 
 #    save this script to the directory as sd_fp16.py. Modify the root_dir if needed.
 #    python sd_fp16.py
    
#import os
#import shutil
import onnx
from onnxruntime.transformers.optimizer import optimize_model


for name in ["kp_detector", "generator"]:
    onnx_model_path = f"{name}.onnx"

    # The following will fuse LayerNormalization and Gelu. Do it before fp16 conversion, otherwise they cannot be fused later.
    # Right now, onnxruntime does not save >2GB model so we use script to optimize unet instead.
    m = optimize_model(
        onnx_model_path,
        model_type="bert",
        num_heads=0,
        hidden_size=0,
        opt_level=0,
        optimization_options=None,
        use_gpu=False,
    )
    '''
    # Use op_bloack_list to force some operators to compute in FP32.
    # TODO: might need some tuning to add more operators to op_bloack_list to reduce accuracy loss.
    if name == "safety_checker":
        m.convert_float_to_float16(op_block_list=["Where"])
    else:
        m.convert_float_to_float16(op_block_list=["RandomNormalLike"])
    '''
    m.convert_float_to_float16()
    # Overwrite existing models. You can change it to another directory but need copy other files like tokenizer manually.
    optimized_model_path = f"{name}_fp16.onnx"
    onnx.save_model(m.model, optimized_model_path)