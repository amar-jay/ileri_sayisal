To run this project 

1. install  package dependencies using(using pip and apt)
    ```bash
    pip install -r requirements.txt
    make install
    ```

2. download dataset. There are **two different methods**  
    - **from google drive** (need client_secret.json from Abdel-manan first.) 
    ```bash
    python3 script/fetch_drive.py
    ```
    - **using torrent** (need debian/ubuntu to run this) 
    ```python
    make fetch_drive
    ```
3. unzip all .nii.zip files `
    ```
    make unzip
    # or python3 script/unzip.py --source_dir=dataset/LITS17 --target_dir=dataset/nii
    ```
4. check if dataset is working right.
    ```
    python3 dataset.py
    ```
5. train model - note there are two models, choose carefully depending on size wanted.
    ```
    python3 train.py -h
    ```
6. quantize trained model `python3 quantize.py`