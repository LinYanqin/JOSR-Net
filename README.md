# JOSR-Net

Official keras implementation for **Jointly Optimized Sampling and Reconstruction (JOSR-Net)**, presented in the paper [JOSR-Net A jointly optimized sampling and reconstruction deep learning network for accelerated NMR spectroscopy]

## Reproduce The Results In The Paper

### Environment

the code is performed in 
    python==3.6
    tensorflow==1.14.0
    scipy==1.5.4
    tensorlayer==1.7.2
    keras==2.2.4
    numpy==1.16.2
    h5py==2.10.0
    nmrglue==0.7
    cuda==10.2

### Download pre-trained model weights

pre-trained model (all the models that appear in the paper) can be download in 'https://www.dropbox.com/scl/fo/t17im9easxyvc1hsy6alj/AHLFUS4w-vTdGzon1_OgjlM?rlkey=740bzlms09f29o989axwklgcs&st=v7or8zsx&dl=0', and put it into `1D/JOSR_model/` or `2D/JOSR_model/`.


### For 1D model
`inference.py` is used to generate sampling scheme or reconstruct undersampling data.
The detailed options in `inference.py` are following:
- `version` is the model path.
- `data_index` is the data path to store the sampling scheme and reconstructed data, and is the path to load the undersampling data, and its definition is `test/data_index/`.
- `rec` is whether to reconstruct, if `rec == False`, `inference.py` is only to generate sampling scheme. 
- `train_M` is the direct dimension of the data.
- `fid_cols` is the indirect dimension of the data to input the network.
- `mux_out` is the samping point of the data.
- `DPS_cols` is the indirect dimension of the data.

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
- `DPS_rows and DPS_cols` is the indirect dimension of the data.

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
- `DPS_cols` is the indirect dimension of the data.
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
- `DPS_rows and DPS_cols` is the indirect dimension of the data.
- `data_path` is the path of 2D dataset.

## Key Workflow Sequence
### For 1D model
1. Set `rec=False` in the `inference.py` file, and run `inference.py`. This will generate a sampling mask.
2. Prerequisite: Place your NMR data file (either a fully sampled FID in nmrPipe format or an already zero-filled undersampled FID) in the directory `test/BMRB_data/`.
3. Preprocess NMR data:
   1. Navigate to the data directory: `cd test/BMRB_data/`.
   2. Modify the `ft1.com` file: Adjust the direct dimension's phase correction and chemical shift range as needed.
   3. Preprocess the data: Execute the commands `ft1.com` and `ft2.com`.
4. Convert to MATLAB format: Navigate back to the main directory (`cd ../..`) and run `nmrPipe_to_mat.py` to convert the processed nmrPipe file into MATLAB files.
5. Prepare Network Input Data (Choose one option based on data type):
   Option A (Fully Sampled Data):
     1. Run `Matlab_process/rec_BMRB.m` in MATLAB.
     2. This script will:
       - Perform undersampling.
       - Generate the required network input files:
         - `input_data.mat`: Undersampled data (shape:[direct_dim_size, 1, indirect_dim_size, 2] - real and imag dual-channel)
         - `factor.mat`: Normalization factor data (shape: [1, direct_dim_size])
         - `mask.mat`: Sampling mask data (shape:[direct_dim_size, 1. indirect_dim_size])
   Option B (Undersampled Data):
     1. Run `Matlab_process/rec_nus.m` in MATLAB.
     2. This script will convert the undersampled data into the same network input format (`input_data.mat`,`factor.mat`,`mask.mat`).
6. Set `rec=True` in the `inference.py` file, and run `inference.py`. This will reconstruct the undersampled data.
7. View Results: Run `Matlab_process/test_BMRB.m` in MATLAB to view the reconstruction results.
   - If you started with undersampled data (Option B in Step 5), you must modify the `filename` in `Matlab_process/test_BMRB.m` to point to the corresponding fully sampled data (Matlab format) file path, or comment out the first 8 lines of `Matlab_process/test_BMRB.m`.
### For 2D model
1. Set `rec=False` in the `inference_4channel.py` file, and run `inference_4channel.py`. This will generate a sampling mask.
2. Prerequisite: Place your NMR data file (either a fully sampled FID in nmrPipe format or an already zero-filled undersampled FID) in the directory `Processed_data/BMRB_data/`.
3. Preprocess NMR data:
   1. Navigate to the data directory: `cd Processed_data/BMRB_data/`.
   2. Modify the `process_direct.com` file: Adjust the direct dimension's phase correction and chemical shift range as needed.
   3. Preprocess the data: Execute the commands `process_direct.com` and `process_zerofill.com`.
4. Convert to MATLAB format: Navigate back to the main directory (`cd ../..`) and run `nmrPipe_to_mat.py` to convert the processed nmrPipe file into MATLAB files.
5. Prepare Network Input Data (Choose one option based on data type):
   Option A (Fully Sampled Data):
     1. Run `Matlab_process/preprocess_BMRB.m` in MATLAB.
     2. This script will:
       - Perform undersampling.
       - Generate the required network input files:
         - `inputreal1.mat`: Undersampled data (shape:[direct_dim_size, indirect_dim_size1, indirect_dim_size2, 2] - R1R2 and R1I2)
         - `inputreal2.mat`: Undersampled data (shape:[direct_dim_size, indirect_dim_size1, indirect_dim_size2, 2] - I1R2 and I1I2)
         - `factor1.mat`: Normalization factor data for `inputreal1.mat` (shape: [1, direct_dim_size]) 
         - `factor2.mat`: Normalization factor data for `inputreal2.mat` (shape: [1, direct_dim_size])
         - `mask3D.mat`: Sampling mask data (shape:[direct_dim_size, indirect_dim_size1. indirect_dim_size2])
   Option B (Undersampled Data):
     1. Run `Matlab_process/preprocess_nus.m` in MATLAB.
     2. This script will convert the undersampled data into the same network input format (`inputreal1.mat`,`inputreal2.mat`,`factor1.mat`,`factor2.mat`,`mask3D.mat`).
6. Set `rec=True` in the `inference_4channel.py` file, and run `inference_4channel.py`. This will reconstruct the undersampled data.
7. View Results: 
   - Run `labelFT.com` and `nmrPipe_to_mat.py` to convert the fully sampled FID nmrPipe file into MATLAB file, and run `Matlab_process/complute_RLNE.m` in MATLAB to view the reconstruction results.
   - If you started with undersampled data (Option B in Step 5), you must place the corresponding fully sampled data (Matlab format) in `Res_data/BMRB_data/`, or comment out the 23-32 lines of `Matlab_process/complute_RLNE.m`.