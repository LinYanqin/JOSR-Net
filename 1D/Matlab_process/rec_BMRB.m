clear;
filename = '../test/BMRB_data/';
filepath = '../test/BMRB_data/';

label = load(strcat(filename,'test.mat'));
label = label.data;
label = fliplr(label);
[M,N] = size(label);
label_f = label/max(real(label(:)));
t_label = ifft(label_f,N,2);

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
    [f,input_i] = saveUnderSampledSpectrumToTXT(FID2,1,1);
    label_i = saveSpectrumToTXT(FID1,1,1,f);
    input_data(i,1,:,1) = input_i(:,1);
    input_data(i,1,:,2) = input_i(:,2);
    label_data(i,1,:,1) = label_i(:,1);
    label_data(i,1,:,2) = label_i(:,2);
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