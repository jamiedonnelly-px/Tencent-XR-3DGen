from onnxsim import simplify
import onnx

output_path = "raftstereo-realtime_regModify_float32_op11.onnx"
output_path1 = output_path.replace('.onnx','_simple.onnx')
onnx_model = onnx.load(output_path)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path1)
print('finished exporting onnx')
