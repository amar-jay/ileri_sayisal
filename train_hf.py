import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import SamModel, SamProcessor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from typing import Dict, Tuple

class SAMFineTuner:
    def __init__(
        self,
        model_name: str = "facebook/sam-vit-huge",
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        num_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = SamModel.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Freeze encoder
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False
            
        # Initialize wandb
        wandb.init(project="sam-lits-finetuning")
        
    def compute_loss(self, pred_masks: torch.Tensor, true_masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between predicted and true masks.
        Combines Binary Cross Entropy and Dice Loss.
        """
        bce_loss = nn.BCEWithLogitsLoss()
        pred_masks = pred_masks.squeeze(1)  # Remove channel dimension
        true_masks = true_masks.float()
        
        # BCE Loss
        bce = bce_loss(pred_masks, true_masks)
        
        # Dice Loss
        pred_probs = torch.sigmoid(pred_masks)
        numerator = 2 * torch.sum(pred_probs * true_masks, dim=(1, 2))
        denominator = torch.sum(pred_probs + true_masks, dim=(1, 2))
        dice = 1 - (numerator + 1) / (denominator + 1)
        dice = torch.mean(dice)
        
        return bce + dice
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Train the model on the provided data.
        """
        optimizer = AdamW(self.model.mask_decoder.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}') as pbar:
                for batch in pbar:
                    pixel_values = batch['pixel_values'].to(self.device)
                    true_masks = batch['mask'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(pixel_values=pixel_values)
                    pred_masks = outputs.pred_masks
                    
                    # Compute loss
                    loss = self.compute_loss(pred_masks, true_masks)
                    train_loss += loss.item()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pth')
                
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss
                })
                print("The training loss: ", train_loss/len(train_loader), "the validation loss: ", val_loss)
            scheduler.step()
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on validation data.
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                true_masks = batch['mask'].to(self.device)
                
                outputs = self.model(pixel_values=pixel_values)
                pred_masks = outputs.pred_masks
                
                loss = self.compute_loss(pred_masks, true_masks)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)



if __name__ == "__main__":
    from dataset import LITSDataset, LITSImageTransform

    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    test_size = 0.2
    mapping_cache_path = "slice_mapping_sam.pkl"
    print("loading 1...")
    train_dataset = LITSDataset(
        images_dir="dataset/nii",
        masks_dir="dataset/nii",
        slice_axis=2,
        mapping_cache_path=mapping_cache_path,
        transform=LITSImageTransform(
            processor = processor,
        ),
        test_size=test_size,
        split="train")
    print("loading 2...")
    val_dataset = LITSDataset(
        images_dir="dataset/nii",
        masks_dir="dataset/nii",
        slice_axis=2,
        mapping_cache_path=mapping_cache_path,
        transform=LITSImageTransform(
            processor = processor,
        ),
        test_size=test_size,
        split="test")
        
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize and train model
    finetuner = SAMFineTuner(
        model_name="facebook/sam-vit-huge",
        learning_rate=1e-5,
        batch_size=8,
        num_epochs=10
    )
    
    finetuner.train(train_loader, val_loader)