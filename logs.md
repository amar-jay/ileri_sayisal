## 1. Test 1 - $18/2/2025$
```sh
⚡ main ~/ileri_sayisal python3 dataset.py
dict_keys(['image', 'mask', 'image_path', 'slice_idx'])
dataset testing...
dataset=<__main__.LITSDataset object at 0x7f85655ae800>
len(dataset)=8825
dataloader testing...
dataloader=<torch.utils.data.dataloader.DataLoader object at 0x7f85656b3e50>
TIME TO  LOAD  DATASET =  227.5285s
TIME TO SAMPLE DATASET =  5.9275s
```

**NOTE:** it takes too long to load dataset, and too long to sample dataset. if there are 8825 samples 
it means it takes roughly $52310$ s or $15$ hours to sample a dataset, not to talk of time to train. 
These values is unreasonable. 

NEXT STEP: try to see if you can sample directly from GPU.

----

## 2. Test 2 - $19/2/2025$
