import os
import nibabel as nib
import zipfile
from tqdm.notebook import trange


def unzip_dir(source_dir, target_dir):
    NO_ERRORS = 0
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    # Iterate through all files in the source directory and its subdirectories
    for root, _, files in os.walk(source_dir):
        for i in trange(len(files)):
            if files[i].endswith(".nii.zip"):
                # Construct the full path of the zip file and the expected nii file
                zip_file_path = os.path.join(root, files[i])
                nii_file_name = files[i][:-4]  # Remove the .zip extension
                nii_file_path = os.path.join(target_dir, nii_file_name)


                # Check if the nii file already exists in the target directory
                if not os.path.exists(nii_file_path):
                    # Unzip the file if the nii file does not exist
                    try:
                        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                            zip_ref.extractall(target_dir)
                        # check if nii file is working
                        nii_file = nib.load(nii_file_path)
                        _ = nii_file.get_fdata()
                        print(f"Unzipped {target_dir}")
                    except Exception as e:
                        # remove nii file if it exists
                        if os.path.exists(nii_file_path):
                            os.remove(nii_file_path)
                        NO_ERRORS += 1
                        # print(f"Error - {zip_file_path}: {e}")

    NO_VALID = len(os.listdir(target_dir))
    print(f"NO OF CORRUPTED FILES: {NO_ERRORS} {NO_ERRORS/(NO_VALID + NO_ERRORS)*100:.4f}%")

if __name__ == "__main__":

    # Define the source and target directories
    import argparse
    parser = argparse.ArgumentParser(description="Unzip a directory to another location.")
    parser.add_argument("source_dir", type=str,default="dataset/zip", help="Path to the source directory.")
    parser.add_argument("target_dir", type=str,  default="dataset/nii", help="Path to the target directory.")

    args = parser.parse_args()
    unzip_dir(args.source_dir, args.target_dir)
