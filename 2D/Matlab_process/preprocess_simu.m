clear;
filename = "../Processed_data/EM_T/test64.mat";
filepath='../Processed_data/EM_T/EM/5e-4/';
data = load(filename);
data = data.FID_zf;
fid = data;
[y_axis,x_axis,z_axis]=size(data);
y_axis = y_axis / 2;
x_axis = x_axis / 2;
R1R2 = fid(1:2:end,1:2:end,:);
R1I2 = fid(1:2:end,2:2:end,:);
I1R2 = fid(2:2:end,1:2:end,:);
I1I2 = fid(2:2:end,2:2:end,:);
FID_real1 = R1R2+1j*R1I2;
FID_real2 = I1R2+1j*I1I2; %Recombine data

load(strcat(filepath,'DPSmask.mat'));
mask=DPSmask;

input_real1 = zeros(z_axis,y_axis,x_axis,2);
input_real2 = zeros(z_axis,y_axis,x_axis,2);
factor1 = zeros(1,z_axis);
factor2 = zeros(1,z_axis);
for i = 1:z_axis
    FID1 = FID_real1(:,:,i);
    FID1_under = FID1.*mask;
    FID2 = FID_real2(:,:,i);
    FID2_under = FID2.*mask;
    [f,under] = saveUnderSampledSpectrumToTXT(FID1_under,i,1);
    factor1(1,i) = f;
    input_real1(i,:,:,:) = under;
    [f,under] = saveUnderSampledSpectrumToTXT(FID2_under,i,1);
    factor2(1,i) = f;
    input_real2(i,:,:,:) = under;
end
label_name = strcat(filepath,'label_3D.mat');
input_name1 = strcat(filepath,'inputreal',num2str(1),'.mat');
input_name2 = strcat(filepath,'inputreal',num2str(2),'.mat');
factor_name1 = strcat(filepath,'factor',num2str(1),'.mat');
factor_name2 = strcat(filepath,'factor',num2str(2),'.mat');
mask_name = strcat(filepath,'mask3D.mat');

save(input_name1,'input_real1');
save(input_name2,'input_real2');
save(factor_name1,'factor1');
save(factor_name2,'factor2');
save(mask_name,'mask');
save(label_name,'fid');