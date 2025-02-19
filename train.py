import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool2d(e3, kernel_size=2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2))
        
        # Decoder
        d4 = self.dec4(torch.cat([F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))
        
        # Final output
        out = self.final_conv(d1)
        return out

def get_model_small():
    """
    This assumes num_classes =2
    """
    model = UNet()
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


def train_model_small(model, dataloader, criterion, optimizer, num_epochs=3, device="cpu"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for sample in tqdm(dataloader, leave=False, desc=f"Loss: {epoch_loss / len(dataloader):.4f}"):
            images = sample['image']
            masks = sample['mask']
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            return
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

def train_model_large(model, dataset, criterion, optimizer, num_epochs=3, batch_size=4, device="cpu"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for sample in tqdm(dataloader, desc=f"model large training..."):
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


if __name__ == "__main__":
    from dataset import LITSDataset, LITSImageTransform
    import argparse
    import shutil
    import os

    parser = argparse.ArgumentParser(description="Download directory from google drive")
    parser.add_argument("--build_path", type=str,default='build', help="Path where trained model is stored")
    parser.add_argument("-d", "--device", type=str,default='cpu', help="Device to train on")
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
    test_size=0.2,
    split="train")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("length of dataset = ", len(dataset),  "\n", "-"*8)
    if args.use_small == True:
        print("Testing small model")
        model = get_model_small()
        input_tensor = torch.rand(4, 1, 512, 512)
        input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Repeat channels for a 3-channel input
        output = model(input_tensor)
        print("output=", output.squeeze(1).shape)
        print("Start training...")
        train_model_small(model, dataloader, nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters(), lr=0.001), num_epochs=args.epochs)
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
        train_model_large(
            model=model, 
            dataset=dataset,
            criterion= nn.CrossEntropyLoss(), 
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.001), 
            num_epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size
            )


    # save the trained model
    print("Saving model in ", args.build_path)
    if os.path.exists(save_path):
        os.remove(save_path)
    torch.save(model.state_dict(), save_path) 
    print('Trained model written to',save_path)
    print("Finished training successfully")
