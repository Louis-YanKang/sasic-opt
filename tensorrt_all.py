
import tensorrt as trt
import cv2 
import numpy as np
import torch
import pickle

from torchvision import transforms

logger = trt.Logger(trt.Logger.INFO)
with open("wrapper_fp32.trt", "rb") as f, trt.Runtime(logger) as runtime:
    engine=runtime.deserialize_cuda_engine(f.read())


'''
for idx in range(engine.num_bindings):
    is_input = engine.binding_is_input(idx)
    name = engine.get_binding_name(idx)
    op_type = engine.get_binding_dtype(idx)
    #model_all_names.append(name)
    shape = engine.get_binding_shape(idx)

    print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)
'''

def trt_version():
    return trt.__version__


def torch_version():
    return torch.__version__


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


image_lf = cv2.imread("assets/frankfurt_000000_009291_leftImg8bit.png")
image_rt = cv2.imread("assets/frankfurt_000000_009291_rightImg8bit.png")
image_lf = cv2.resize(image_lf, (896,352))
image_rt = cv2.resize(image_rt, (896,352))
image_lf = image_lf.transpose(2,0,1)
image_rt = image_rt.transpose(2,0,1)
img_input_lf = image_lf.astype(np.float32)
img_input_rt = image_rt.astype(np.float32)
img_input_lf = torch.from_numpy(img_input_lf)
img_input_rt = torch.from_numpy(img_input_rt)
img_input_lf = img_input_lf.unsqueeze(0)
img_input_rt = img_input_rt.unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_input_lf = img_input_lf.to(device)
img_input_rt = img_input_rt.to(device)



class TRTModule(torch.nn.Module):
    def __init__(self, engine, input_names, output_names):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine创建执行context
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            # 设定shape
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            #print("self.context.get_binding_shape(idx) == ", self.context.get_binding_shape(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings,torch.cuda.current_stream().cuda_stream)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs



trt_model = TRTModule(engine, ["left", "right"], ["output_left","output_right"])


xl_hat, xr_hat = trt_model(img_input_lf, img_input_rt)


left = transforms.ToPILImage()(xl_hat[0])
right = transforms.ToPILImage()(xr_hat[0])

print(f'  ## Save image as output_left, output_right')
left.save('output_left_fp32.png', 'PNG')
right.save('output_right_fp32.png', 'PNG')



