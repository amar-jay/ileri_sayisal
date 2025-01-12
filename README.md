To run this project 

1. install  package dependencies using `pip install -r requirements.txt`
2. download dataset from google drive (need authentication by Abdel-manan first) `python3 fetch_drive.py`
3. unzip all .nii.zip files `python3 unzip.py`
4. check if dataset is working right `python3 dataset.py`
5. train model - note there are two models, choose carefully depending on size wanted. `python3 train.py -h`
6. quantize trained model `python3 quantize.py`