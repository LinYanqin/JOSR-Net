clear;
filename = '../test/EM_T/test128.mat';
filepath = '../test/EM_T/EM/5e-3/';

data=load(filename);
data=data.FID1;
[M,N] = size(data);
label_f=fft2(data);
label_f = label_f/max(real(label_f(:))); %Pure label spectrum
t_label = ifft(label_f,N,2); %Obtain data with indirect dimensions in the time domain

load(strcat(filepath, 'DPSmask.mat'));
mask=DPSmask;
under = t_label.*mask;

input_data = zeros(M,1,N,2);
label_data = zeros(M,1,N,2);
factor = zeros(1,M);
mask_data = zeros(M,1,N);
for i = 1:M
    FID1 = t_label(i,:);
    FID2 = under(i,:);
    [f,input_gb1] = saveUnderSampledSpectrumToTXT(FID2,1,1);
    label_gb1 = saveSpectrumToTXT(FID1,1,1,f);
    input_data(i,1,:,1) = input_gb1(:,1);
    input_data(i,1,:,2) = input_gb1(:,2);
    label_data(i,1,:,1) = label_gb1(:,1);
    label_data(i,1,:,2) = label_gb1(:,2);
    factor(1,i) = f;
    mask_data(i,1,:) = mask;
end
name_input = strcat(filepath,'input_data.mat');
name_label = strcat(filepath,'label_data.mat');
name_mask = strcat(filepath,'mask.mat');
name_factor = strcat(filepath,'factor.mat');
save(name_input,'input_data');
save(name_label,'label_data');
save(name_mask,'mask_data');
save(name_factor,'factor');
