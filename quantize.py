import os
import sys
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dataset import LITSDataset, LITSImageTransform
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

def test(model, device, test_loader):
    pass
def quantize(model, model_path, quant_model_path, quant_mode,batchsize):
  # load trained model
  model.load_state_dict(torch.load(model_path))

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  rand_in = torch.randn([batchsize, 1, 512, 512])
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model_path) 
  quantized_model = quantizer.quant_model


  dataset = LITSDataset(
    images_dir="/content/dataset/nii",
    masks_dir="/content/dataset/nii",
    slice_axis=2,
    transform=LITSImageTransform(),
    test_size=0.2,
    split="test")

  test_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

  # evaluate 
  test(quantized_model, device, test_loader)


  # export config
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model_path)

  return



def run_main():

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',  '--build_path',  type=str, default='build',    help='Path to build folder. Default is build')
    parser.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    parser.add_argument('-b',  '--batchsize',  type=int, default=4,        help='Testing batchsize - must be an integer. Default is 100')
    parser.add_argument("-b", "--build_path", type=str,default='build', help="Path where trained model is stored")
    parser.add_argument('-s', '--use_small', type=bool,  default=True, help='the size of model, either small or large. Default is True')
    args = parser.parse_args()

    model = None
    model_path = os.path.join(args.build_path, 'f_small_model.pth' if args.use_small else 'f_large_model.pth') # float model path
    q_model_path = os.path.join(args.build_path, 'q_small_model.pth' if args.use_small else 'q_large_model.pth') # float model path

    quantize(
        model=model,
        model_path=model_path,
        quant_model_path=q_model_path,
        quant_mode=args.quant_mode,
        batchsize=args.batchsize
    )

    print(model_path, "model quantized successfully as", q_model_path)
    return



if __name__ == '__main__':
    run_main()

