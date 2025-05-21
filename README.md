# JOSR-Net

Official keras implementation for **Jointly Optimized Sampling and Reconstruction (JOSR-Net)**, presented in the paper [JOSR-Net A jointly optimized sampling and reconstruction deep learning network for accelerated NMR spectroscopy]

## Reproduce The Results In The Paper

### Environment

the code is performed in 
    python == 3.6
    keras == 2.2.4
    tensorflow == 1.14.0
    cuda == 10.2

### Download pre-trained model weights

pre-trained model (all the models that appear in the paper) can be download in 'https://www.dropbox.com/scl/fo/a41xpcu6bvfu7uygcyqw7/AOU7V6UFtrviSIAkvRA5eXs?rlkey=8w2pfceewjrm6y7jxmlu1nez4&st=ic54m96d&dl=0', and put it into `1D/JOSR_model/` or `2D/JOSR_model/`.


### For 1D model
`inference.py` is used to generate sampling scheme or reconstruct undersampling data.
The detailed options in `inference.py` are following:
- `version` is the model path.
- `data_index` is the data path to store the sampling scheme and reconstructed data, and is the path to load the undersampling data, and its definition is `test/data_index/`.
- `rec` is whether to reconstruct, if `rec == False`, `inference.py` is only to generate sampling scheme. 
- `train_M` is the direct dimension of the data.
- `fid_cols` is the indirect dimension of the data to input the network.
- `mux_out` is the samping point of the data.
- `DPS_cols` is the direct dimension of the data.

`Matlab_process/drawmask.m` is used to show sampling scheme.
The detailed options in `Matlab_process/drawmask.m` are following:
- `M` is the indirect dimension of the data.
- `filepath` is the path of sampling scheme.

`Matlab_process/rec_BMRB.m` is used to undersampling data, and `Matlab_process/test_BMRB.m` is used to show reconstructed results.
The detailed options in `Matlab_process/rec_BMRB.m` and `Matlab_process/test_BMRB.m` are following:
- `filename` is the full sampling data.
- `filepath` is the path of sampling scheme, and is the path to store undersampling data or reconstructed data.
`Matlab_process/rec.m` and `Matlab_process/test.m` is used to process simulated data.

### For 2D model
`inference_4channel.py` is used to generate sampling scheme or reconstruct undersampling data.
The detailed options in `inference_4channel.py` are following:
- `version` is the model path.
- `data_index` is the data path to store the sampling scheme and reconstructed data, and is the path to load the undersampling data, and its definition is `Processed_data/data_index/`.
- `rec` is whether to reconstruct, if `rec == False`, `inference_4channel.py` is only to generate sampling scheme. 
- If `rec == True`, `nmrpipe_dat_path` is the path of nmrpipe_data need to be set, which is used to transform mat_data to nmrpiep_data, and `com_path` is the path of nmrpipe script need to be set, which is used to preform nmrpipe operation,
- `train_M` is the direct dimension of the data.
- `fid_rows and fid_cols` is the indirect dimension of the data to input the network.
- `mux_out` is the samping point of the data.
- `DPS_rows and DPS_cols` is the direct dimension of the data.

`Matlab_process/drawmask.m` is used to show sampling scheme.
The detailed options in `Matlab_process/drawmask.m` are following:
- `DPS_rows and DPS_cols` is the indirect dimension of the data.
- `filepath` is the path of sampling scheme.

`Matlab_process/preprocess_BMRB.m` is used to undersampling data, and `Matlab_process/complute_RLNE.m` is used to show reconstructed results.
The detailed options in `Matlab_process/preprocess_BMRB.m` and `Matlab_process/complute_BMRB.m` are following:
- `filename` is the full sampling data.
- `filepath` is the path of sampling scheme, and is the path to store undersampling data or reconstructed data.
`Matlab_process/preprocess_simu.m` and `Matlab_process/complute_simu.m` is used to process simulated data.

## Training 
If you want to train your model with different indirect dimension size or sampling rate. 
For 1D model, you need run `Matlab_process/generate1D.m` to generate 1D dataset, and run `data_loader.py` to divide the dataset into a training set and a validation set. Then, you can run `main.py` to train your model in this dataset.
For 2D model, you need run `Matlab_process/generate2D.m` to generate 2D dataset, and run `data_loader.py` to divide the dataset into a training set and a validation set. Then, you can run `main_4channel.py` to train your model in this dataset.

### For 1D model
`Matlab_process/generate1D.m` is used to generate 1D dataset.
The detailed options in `Matlab_process/generate1D.m` are following:
- `N1` is the indirect dimension of the data, which similar to `DPS_cols`.
- `N2` is the indirect dimension of the data to input the network, which similar to `fid_cols`.
- `savepath` is the path to store the 1D dataset.
- `maskpath` is the path of the mask, which makes the normalization factor of the full sampling data consistent with that of the undersampling data.

`data_loader.py` is used to divide the dataset into a training set and a validation set.
The detailed option in `data_loader.py` is following:
- `data_path` is the path of 1D dataset.

`main.py` is used to train JOSR-Net model.
The detailed options in `main.py` are following:
- `fid_cols` is the indirect dimension of the data to input the network.
- `mux_out` is the samping point of the data.
- `DPS_cols` is the direct dimension of the data.
- `data_path` is the path of 1D dataset.

### For 2D model
`Matlab_process/generate2D.m` is used to generate 2D dataset.
The detailed options in `Matlab_process/generate2D.m` are following:
- `N1` is the indirect dimension 1 of the data, which similar to `DPS_rows`.
- `N2` is the indirect dimension 2 of the data, which similar to `DPS_cols`.
- `savepath` is the path to store the 2D dataset.
- `maskpath` is the path of the mask, which makes the normalization factor of the full sampling data consistent with that of the undersampling data.

`data_loader.py` is used to divide the dataset into a training set and a validation set.
The detailed option in `data_loader.py` is following:
- `data_path` is the path of 2D dataset.

`main_4channel.py` is used to train JOSR-Net model.
The detailed options in `main_4channel.py` are following:
- `fid_rows and fid_cols` is the indirect dimension of the data to input the network.
- `mux_out` is the samping point of the data.
- `DPS_rows and DPS_cols` is the direct dimension of the data.
- `data_path` is the path of 1D dataset.


# References 
If you find this repository useful for your research, please cite the following work.
```
@article{
  title={JOSR-Net A jointly optimized sampling and reconstruction deep learning network for accelerated NMR spectroscopy},
  author={Chen, W. H. et al.},
}
```
This implementation is based on / inspired by:
- https://github.com/IamHuijben/Deep-Probabilistic-Subsampling (DPS)

