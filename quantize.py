import os
import sys
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from dataset import LITSDataset, LITSImageTransform
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from train import get_model_large, get_model_small
from test import evaluate
def test(model, batchsize, device="cpu"):
  dataset = LITSDataset(
    images_dir="./dataset/",
    masks_dir="./dataset/",
    slice_axis=2,
    transform=LITSImageTransform(),
    test_size=0.5,
    split="all"
  )

  acc= evaluate(model, dataset, batch_size=batchsize, device=device) 

  print(f"Quantized Model accuracy is {acc*100:.4f}%")

def quantize(model_path, quant_model, quant_mode,batchsize, device="cpu", use_small=False):
  # load trained model
  model = None
  if use_small:
    model = get_model_small()
  else:
    model = get_model_large(3, model_path, device=device)

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  rand_in = torch.randn([4, 3, 512, 512])
  quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
  quantized_model = quantizer.quant_model
  print("quantized.........")
  # evaluate 
  test(quantized_model, batchsize)


  # export config
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

  return



def run_main():
    from train import get_model_small
    import os
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True,    help='Path to build folder. Default is build')
    parser.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    parser.add_argument('-b',  '--batchsize',  type=int, default=4,        help='Testing batchsize - must be an integer. Default is 4')
    parser.add_argument('-s', '--use_small', action='store_true', 
                    help='Use small model size (default: True). Pass -s to use large model.')
    args = parser.parse_args()

    model_path = args.model_path
    q_model_path = os.path.join(os.path.dirname(model_path), 'quantized') # float model path

    assert os.path.exists(model_path)
    os.makedirs(q_model_path, exist_ok=True)

    quantize(
        model_path=model_path,
        quant_model=q_model_path,
        quant_mode=args.quant_mode,
        batchsize=args.batchsize,
        use_small=args.use_small
    )

    print(model_path, "model quantized successfully in", q_model_path)
    return



if __name__ == '__main__':
    run_main()

