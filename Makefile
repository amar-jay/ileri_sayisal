install:
	pip install -r requirements.txt
	sudo apt-get install -y aria2
fetch_drive:
	sudo apt-get install -y aria2
	mkdir -p dataset
	wget -O my.torrent https://academictorrents.com/download/27772adef6f563a1ecc0ae19a528b956e6c803ce.torrent
	aria2c --dir=./dataset --seed-time=0 my.torrent
	rm my.torrent # clear torrent after use

unzip:
	python3 script/unzip.py --source_dir=dataset/LITS17 --target_dir=dataset/nii

test_dataset:
	python3 dataset.py 