import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INTERNAL_ERROR)
def build_engine(onnx_path, shape = [1,112,112,3]):

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
      builder_config = builder.create_builder_config()
      builder_config.max_workspace_size = 1 << 32
      builder.max_batch_size = 32


      builder_config.set_flag(trt.BuilderFlag.FP16)
      builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
      # builder.platform_has_fast_fp16 = True
      # builder.platform_has_fast_int8  = True

   #    builder.max_workspace_size = (256 << 20)
      with open(onnx_path, 'rb') as model:
         parser.parse(model.read())
      network.get_input(0).shape = shape
   #    engine = builder.build_cuda_engine(network)
      plan = builder.build_serialized_network(network, builder_config)
   #    engine = runtime.deserialize_cuda_engine(plan)
      with trt.Runtime(TRT_LOGGER) as runtime:
         engine = runtime.deserialize_cuda_engine(plan)
      return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)
def load_engine(trt_runtime, engine_path):
   with open(engine_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine
