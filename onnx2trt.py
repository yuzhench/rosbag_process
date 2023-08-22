import tensorrt as trt

def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        parser.parse(model.read())

    config.max_workspace_size = 1 << 31

    profile = builder.create_optimization_profile()
    profile.set_shape('input.1', (1, 3, 512, 512), (1, 3, 512, 512), (1, 3, 512, 512))
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    if engine is None:
        print('Failed to create engine')
        return None

    return engine



def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)

def main():
    engine = build_engine("model.onnx")
    save_engine(engine, "trt_engine.trt")

if __name__ == "__main__":
    main()
