from rknn.api import RKNN

path_to_onnx = './onnxs/train14_411.onnx'
path_to_dataset = './samples/227/227.txt'
path_to_export = './rknns/train14_411.rknn'

rknn = RKNN(verbose=True, verbose_file='./mobilenet_build.log')

rknn.config(
    mean_values=[[130.7362084308155, 115.92932719021557, 112.76286192868483]],
    std_values=[[50.71546470866305, 61.64713363321818, 67.17437275240536]],
    quant_img_RGB2BGR=False,
    # quantized_dtype=asymmetric_quantized-8,
    quantized_algorithm='normal',
    quantized_method='channel',
    # float_dtype=float16,
    optimization_level=1,
    target_platform="rk3588",
    model_pruning=True)

ret = rknn.load_onnx(model = path_to_onnx)
if ret != 0:
    print("模型导入失败")
    exit(1)

ret = rknn.build(do_quantization=True, dataset=path_to_dataset)
if ret != 0:
    print("模型构建失败")
    exit(1)

ret = rknn.export_rknn(export_path = path_to_export)
if ret != 0:
    print("模型导出失败")
    exit(1)

rknn.release()

