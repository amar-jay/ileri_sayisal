import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, LRASPP_MobileNet_V3_Large_Weights

# Model Definition
def get_model_small(num_classes, weights_path, device="cpu"):
    model = models.segmentation.lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    in_channels_low = model.classifier.low_classifier.in_channels  # should be 40
    model.classifier.low_classifier = nn.Conv2d(in_channels_low, num_classes, kernel_size=1)

    # Replace high_classifier
    in_channels_high = model.classifier.high_classifier.in_channels  # should be 128
    model.classifier.high_classifier = nn.Conv2d(in_channels_high, num_classes, kernel_size=1)

    model.to(device)
    
    # Check if weights exist locally    
    if os.path.exists(weights_path):
        print(f"Loading weights from local directory: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    return model

# Model Definition
def get_model_large(num_classes, weights_path, device="cpu"):
    model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.to(device)
    
    # Check if weights exist locally    
    if os.path.exists(weights_path):
        print(f"Loading weights from local directory: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    return model


def train_model(model, dataset, criterion, optimizer, save_path, num_epochs=3, batch_size=4, device="cpu"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for sample in tqdm(dataloader, desc=f"training model..."):
            images = sample['image'].to(device) # Shape: [B, 1, 512, 512]
            masks = sample['mask'].to(device)   # Shape: [B, 512, 512]
            
            images = images.repeat(1, 3, 1, 1)  # Repeat channels for a 3-channel input
            images_expected_shape = (batch_size, 3, 512, 512)
            masks_expected_shape = (batch_size, 512, 512)
            if masks.ndim  == 4:
                masks = masks.squeeze(1)

            # NOTE: BAD WAY OF RESOLVING THE ISSUE, FIND THE PROBLEM FIRST.
            if images.shape != images_expected_shape or masks.shape != masks_expected_shape:
                print(f"There is an issue to the image or mask shape\n{images.shape=} expected {images_expected_shape=}\n{masks.shape=} expected  {masks_expected_shape=}")
                continue
            # Forward pass
            outputs = model(images)['out']

            # NOTE: 
            # DONT KNOW IF THIS IS THE BEST WAY TO DO THIS BUT MASK GIVES 0, 0.0039, and 0.0078 consistently in this dataset.
            # I  AM RESCALING THE DATASET TO THE 0.0078
            masks /= masks.max()
            # print(outputs.dtype, masks.dtype, torch.unique(masks), torch.unique(outputs))
            loss = criterion(outputs, masks.long())
            # in inference
            #outputs = torch.argmax(outputs, dim=1, keepdim=True).squeeze(1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        print("DONE. - ", epoch)
        print("Saving model in ", args.build_path)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(model.state_dict(), save_path) 


if __name__ == "__main__":
    from dataset import LITSDataset, LITSImageTransform
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Download directory from google drive")
    parser.add_argument("--build_path", type=str,default='build', help="Path where trained model is stored")
    parser.add_argument("-d", "--device", type=str,default='cuda', help="Device to train on")
    parser.add_argument('-e', '--epochs', type=int,  default=3, help='Number of training epochs. Must be an integer. Default is 3')
    parser.add_argument("-b", '--batch_size', type=int,  default=4, help='Number of train batches. Must be an integer. Default is 4')
    parser.add_argument('-s', '--use_small', action='store_true', 
                    help='Use small model size (default: True). Pass -s to use large model.')

    args = parser.parse_args()
    print("\n", args, "\n", "-"*8)

    # weights path    
    os.makedirs(args.build_path, exist_ok=True)
    save_path = os.path.join(args.build_path, 'f_small_model.pth' if args.use_small else 'f_large_model.pth') # float model path

    dataset = LITSDataset(
    images_dir="dataset/nii",
    masks_dir="dataset/nii",
    slice_axis=2,
    num_channels=1,
    transform=LITSImageTransform(),
    test_size=0.05,
    split="train")
    dataset.set_split("train")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("length of dataset = ", len(dataset),  "\n", "-"*8)


    if args.use_small == True:
        print("Testing small model")
        model = get_model_small(3, save_path)
        input_tensor = torch.rand(4, 1, 512, 512)
        input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Repeat channels for a 3-channel input
        output = model(input_tensor)['out']
        output = torch.argmax(output, dim=1, keepdim=True)
        output = output.squeeze(1)
        print("output=", output.shape)
    else:
        print("Testing large model")
        model = get_model_large(3, save_path)
        input_tensor = torch.rand(4, 1, 512, 512)
        input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Repeat channels for a 3-channel input
        output = model(input_tensor)['out']
        output = torch.argmax(output, dim=1, keepdim=True)
        output = output.squeeze(1)
        print("output=", output.shape)
    
    print("Start training...")
    train_model(
        model=model, 
        dataset=dataset,
        criterion= nn.CrossEntropyLoss(), 
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001), 
        num_epochs=args.epochs,
        save_path=save_path,
        device=args.device,
        batch_size=args.batch_size
        )


    # save the trained model
    print('Trained model written to',save_path)
    print("Finished training successfully")
