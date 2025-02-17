import torch 
import sys
import argparse
from collections import OrderedDict
sys.path.append('core')
from raft_stereo import RAFTStereo

#Function to Convert to ONNX 
def Convert_ONNX(args): 

    model = RAFTStereo(args)
    # model.load_state_dict(torch.load(args.restore_ckpt))

    state_dict = torch.load(args.restore_ckpt)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if "fnet" not in k:
            name = k.replace('module.','')# remove `module.`
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to('cuda')
    model.eval() 

    # for module in model.modules():
    #     if isinstance(module, torch.nn.modules.instancenorm._BatchNorm):
    #         module.track_running_stats = False
    #         module.running_var = None
    #         module.running_mean = None

    # Let's create a dummy input tensor  
    input_size1 = (3,256,320)
    dummy_input1 = torch.randn(1, *input_size1).cuda()
    dummy_input2 = torch.randn(1, *input_size1).cuda()

    # out = model(dummy_input1, dummy_input2, iters=args.valid_iters, test_mode=True)
    # print(len(out))
    # print(out[0].shape)
    # print(out[1].shape)
    # Export the model   
    model_output_name = args.restore_ckpt.split('/')[-1].replace('.pth','_regModify_float32_op11.onnx')
 
    torch.onnx.export(model,         # model being run 
         (dummy_input1,dummy_input2),       # model input (or a tuple for multiple inputs) 
         model_output_name,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         training=torch.onnx.TrainingMode.EVAL,
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         verbose=False,
         input_names = ["Input1","Input2"],   # the model's input names 
         output_names = ["Output"])
    print('Model has been converted to ONNX')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=6, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    Convert_ONNX(args)