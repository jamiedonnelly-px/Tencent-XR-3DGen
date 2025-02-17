import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
 

class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        scores, boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores, boxes



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

model_path="raftstereo-realtime_regModify_float32_op11_simple.onnx"

img1_path="input1_nchw.raw"
img2_path="input2_nchw.raw"
img1 = np.fromfile(img1_path, dtype=np.float32)
img1 = np.reshape(img1,(3,256,320))
img2 = np.fromfile(img2_path, dtype=np.float32)
img2 = np.reshape(img2,(3,256,320))

img1_nhwc = torch.from_numpy(img1).permute(1,2,0).float()
img2_nhwc = torch.from_numpy(img2).permute(1,2,0).float()
to_numpy(img1_nhwc).tofile("input1_nhwc.raw")
to_numpy(img2_nhwc).tofile("input2_nhwc.raw")

img1 = torch.from_numpy(img1).float()
img1 = img1.unsqueeze_(0)
img2 = torch.from_numpy(img2).float()
img2 = img2.unsqueeze_(0)
print(img1.shape)
print(img2.shape)

net_session = onnxruntime.InferenceSession(model_path)
for input in net_session.get_inputs():
    print(input.name)
for output in net_session.get_outputs():
    print(output.name)
inputs = {net_session.get_inputs()[0].name: to_numpy(img1),net_session.get_inputs()[1].name: to_numpy(img2)}
outs = net_session.run(None, inputs)
print(outs[1].shape)
plt.imsave("ziyou_output/output.png", -outs[1][0,0,:,:], cmap='jet')
# out = -outs[1][0,0,:,:]
# out = (out-out.min())/(out.max()-out.min())
# cv2.imwrite("ziyou_output/output.png", out)