import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from typing import Optional, Tuple, Union
from pathlib import Path

def visualize_nii_3d(
    nii_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    slice_nums: Optional[Tuple[int, int, int]] = None,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'gray',
    mask_alpha: float = 0.3,
    title: Optional[str] = "visualization"
) -> None:
    """
    Visualize a NIfTI file with optional mask overlay.

    Args:
        nii_path: Path to the NIfTI file
        mask_path: Optional path to the mask NIfTI file
        slice_nums: Optional tuple of (sagittal, coronal, axial) slice numbers
        figsize: Figure size
        cmap: Colormap for the image
        mask_alpha: Transparency of the mask overlay
        title: Optional title for the plot
    """
   # Load the NIfTI file
    try:
        img = nib.load(str(nii_path))
        data = img.get_fdata()

        # Load the mask if provided
        if mask_path:
            mask = nib.load(str(mask_path))
            mask_data = mask.get_fdata()
        else:
            mask_data = None
    # when file is corrupt
    except OSError:
        print(f"Error loading file: {nii_path}")
        return None

    # Get the middle slice number for each axis if not provided
    if slice_nums is None:
        slice_nums = (
            data.shape[0] // 2,  # sagittal
            data.shape[1] // 2,  # coronal
            data.shape[2] // 2   # axial
        )

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=1.05)

    # Helper function to normalize data for visualization
    def normalize(data):
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min + 1e-8)

    # Plot sagittal view (YZ plane)
    sagittal_slice = normalize(data[slice_nums[0], :, :])
    axes[0].imshow(np.rot90(sagittal_slice), cmap=cmap)
    if mask_data is not None:
        mask_slice = mask_data[slice_nums[0], :, :]
        axes[0].imshow(np.rot90(mask_slice), alpha=mask_alpha, cmap='Reds')
    axes[0].set_title(f'Sagittal (slice {slice_nums[0]})')
    axes[0].axis('off')

    # Plot coronal view (XZ plane)
    coronal_slice = normalize(data[:, slice_nums[1], :])
    axes[1].imshow(np.rot90(coronal_slice), cmap=cmap)
    if mask_data is not None:
        mask_slice = mask_data[:, slice_nums[1], :]
        axes[1].imshow(np.rot90(mask_slice), alpha=mask_alpha, cmap='Reds')
    axes[1].set_title(f'Coronal (slice {slice_nums[1]})')
    axes[1].axis('off')

    # Plot axial view (XY plane)
    axial_slice = normalize(data[:, :, slice_nums[2]])
    axes[2].imshow(np.rot90(axial_slice), cmap=cmap)
    if mask_data is not None:
        mask_slice = mask_data[:, :, slice_nums[2]]
        axes[2].imshow(np.rot90(mask_slice), alpha=mask_alpha, cmap='Reds')
    axes[2].set_title(f'Axial (slice {slice_nums[2]})')
    axes[2].axis('off')

    plt.tight_layout()
    return fig, axes

def visualize_nii(
    nii_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'gray',
    mask_alpha: float = 0.3,
    title: Optional[str] = "visualization"
) -> None:
    """
    Visualize a NIfTI file with optional mask overlay.

    Args:
        nii_path: Path to the NIfTI file
        mask_path: Optional path to the mask NIfTI file
        slice_nums: Optional tuple of (sagittal, coronal, axial) slice numbers
        figsize: Figure size
        cmap: Colormap for the image
        mask_alpha: Transparency of the mask overlay
        title: Optional title for the plot
    """
    # Load the NIfTI file
    try:
        img = nib.load(str(nii_path))
        data = img.get_fdata()

        # Load the mask if provided
        if mask_path:
            mask = nib.load(str(mask_path))
            mask_data = mask.get_fdata()
        else:
            mask_data = None
    # when file is corrupt
    except OSError:
        print(f"Error loading file: {nii_path}")
        return None

    # Get axial slice count
    slice_nums = (
        int(data.shape[2]*0.1),  # from
        int(data.shape[2]*0.5),  # mid
        int(data.shape[2]*0.9),  # tail
    )
    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=1.05)

    # Helper function to normalize data for visualization
    def normalize(data):
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min + 1e-8)

    # Front view (10% plane)
    sagittal_slice = normalize(data[:, :, slice_nums[0]])
    axes[0].imshow(np.rot90(sagittal_slice), cmap=cmap)
    if mask_data is not None:
        mask_slice = mask_data[:, :, slice_nums[0]]
        axes[0].imshow(np.rot90(mask_slice), alpha=mask_alpha, cmap='Reds')
    axes[0].set_title(f'Front (slice {slice_nums[0]})')
    axes[0].axis('off')

    # Mid view (XZ plane)
    coronal_slice = normalize(data[:, :, slice_nums[1]])
    axes[1].imshow(np.rot90(coronal_slice), cmap=cmap)
    if mask_data is not None:
        mask_slice = mask_data[:, :, slice_nums[1]]
        axes[1].imshow(np.rot90(mask_slice), alpha=mask_alpha, cmap='Reds')
    axes[1].set_title(f'Mid (slice {slice_nums[1]})')
    axes[1].axis('off')

    # Tail view (90% plane)
    axial_slice = normalize(data[:, :, slice_nums[2]])
    axes[2].imshow(np.rot90(axial_slice), cmap=cmap)
    if mask_data is not None:
        mask_slice = mask_data[:, :, slice_nums[2]]
        axes[2].imshow(np.rot90(mask_slice), alpha=mask_alpha, cmap='Reds')
    axes[2].set_title(f'Tail (slice {slice_nums[2]})')
    axes[2].axis('off')

    plt.tight_layout()
    return fig, axes

def create_slice_browser(
    nii_path: Union[str, Path],
    axis: int = 2,  # 0: sagittal, 1: coronal, 2: axial
    mask_path: Optional[Union[str, Path]] = None,
    cmap: str = 'gray',
    figsize: Tuple[int, int] = (8, 8)
) -> None:
    """
    Create an interactive slice browser for a NIfTI file.

    Args:
        nii_path: Path to the NIfTI file
        axis: Axis to browse (0: sagittal, 1: coronal, 2: axial)
        mask_path: Optional path to the mask NIfTI file
        cmap: Colormap for the image
        figsize: Figure size
    """
    img = nib.load(str(nii_path))
    data = img.get_fdata()

    if mask_path:
        mask = nib.load(str(mask_path))
        mask_data = mask.get_fdata()
    else:
        mask_data = None

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    class IndexTracker:
        def __init__(self, ax, data, mask_data=None):
            self.ax = ax
            self.data = data
            self.mask_data = mask_data
            self.slc = data.shape[axis] // 2
            self.update()

        def onscroll(self, event):
            if event.button == 'up':
                self.slc = min(self.slc + 1, self.data.shape[axis] - 1)
            else:
                self.slc = max(self.slc - 1, 0)
            self.update()

        def update(self):
            plt.cla()
            if axis == 0:
                img_slice = self.data[self.slc, :, :]
                mask_slice = self.mask_data[self.slc, :, :] if self.mask_data is not None else None
            elif axis == 1:
                img_slice = self.data[:, self.slc, :]
                mask_slice = self.mask_data[:, self.slc, :] if self.mask_data is not None else None
            else:
                img_slice = self.data[:, :, self.slc]
                mask_slice = self.mask_data[:, :, self.slc] if self.mask_data is not None else None

            # Normalize slice
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

            self.ax.imshow(np.rot90(img_slice), cmap=cmap)
            if mask_slice is not None:
                self.ax.imshow(np.rot90(mask_slice), alpha=0.3, cmap='Reds')

            self.ax.set_title(f'Slice {self.slc}/{self.data.shape[axis] - 1}')
            self.ax.axis('off')
            plt.draw()


    tracker = IndexTracker(ax, data, mask_data)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


# Load and visualize both the image and mask NIfTI files
image_path = "dataset/nii/volume-98.nii"
mask_path = "dataset/nii/segmentation-98.nii"

# Basic static visualization
# _ = visualize_nii(
#     nii_path=image_path,
#     mask_path=mask_path,  # Pass the mask NIfTI file here
#     cmap='gray',         # Color map for the main image
#     mask_alpha=0.3       # Transparency of the mask overlay
# )

# # Or use the interactive slice browser
create_slice_browser(
    nii_path=image_path,
    mask_path=mask_path,  # Pass the mask NIfTI file here
    axis=2               # 0: sagittal, 1: coronal, 2: axial
)
