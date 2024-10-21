- dadaGP-generation-conda-env-requirements.txt
	
	python packages required 
	might not be able to conda install everything

- dadaGP-generation-pip-env-requirements.txt

	python packages required
	maybe it's better to cross check this with the packages within "dadaGP-generation-conda-env-requirements.txt"
	it was a long time ago that I set up the environment I'm currently using, but it was painful
	this were the notes I took at the time, in case they are helpful:

		After a morning of struggles, …
		For installation of pytorch-fast-transformers:

		(dadaCP) pps30@bibury:~$ python --version
		Python 3.7.10
		(dadaCP) pps30@bibury:~$ nvcc --version
		nvcc: NVIDIA (R) Cuda compiler driver
		Copyright (c) 2005-2019 NVIDIA Corporation
		Built on Fri_Feb__8_19:08:17_PST_2019
		Cuda compilation tools, release 10.1, V10.1.105
		(dadaCP) pps30@bibury:~$ gcc --version
		gcc (GCC) 5.5.0
		Copyright (C) 2015 Free Software Foundation, Inc.
		This is free software; see the source for copying conditions.  There is NO
		warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
	
		(dadaCP) pps30@bibury:~$ conda list | grep torch
		torch                     1.5.1+cu101              pypi_0    pypi
		torchvision               0.6.1+cu101              pypi_0    pypi


		STILL, only shows when pip list, not when conda list.
		

- full-data-config_5_lat1024.yml
	
	main config file with all the parameters for generation and training

- fulldataset-song-artist-train_data_XL.npz

	dataset in numpy format, for training

- inference_fd5_lat1024.py
	
	script for generation, dependent on:
		- main config file
		- model weights
		- dataset vocabulary and reverse vocabulary

- model_ead.py

	script for model backbone, adapted from this repo:
	https://github.com/YatingMusic/compound-word-transformer

- model weights

	folder containing:
		- model weights (from epoch 200, the best one we have)
		- a config file (mostly useless, but necessary for it to run; can be ignored)

- modules.py

	script for model backbone, from:
	https://github.com/YatingMusic/compound-word-transformer

- rev_vocab_song_artist.pkl

	reverse vocabulary, necessary for training/inference

- train_randomsampling_5_lat1024.py

	script for training, adapted from:
	https://github.com/YatingMusic/compound-word-transformer

	dependent on:
		- main config file
		- dataset vocabulary and reverse vocabulary
		- dataset in numpy format

- vocab_song_artist.pkl

	vocabulary for the dataset, necessary for training/inference