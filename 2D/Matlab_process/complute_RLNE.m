clear;clc;
filepath = '../Res_data/BMRB_data/';

y_axis = 64;  
x_axis = 64;  
z_axis = 280;

res = zeros(1,2*x_axis,z_axis,2*y_axis);
res_CN = zeros(1,2*x_axis,2*y_axis);
max_res = zeros(1,1);
max_res1 = zeros(1,1);
name = strcat(filepath,'resCN3D.mat');
name1 = strcat(filepath,'resCN.mat');
data = load(name);
data = data.resCN3D;
max_res(1,1) = max(data(:));
res(1,:,:,:) = data/max_res(1,1);
data = load(name1);
data = data.resCN;
max_res1(1,1) = max(data(:));
res_CN(1,:,:,:) = data/max_res1(1,1);

res_mean = reshape(res,[2*x_axis,z_axis,2*y_axis]);
resCN = reshape(res_CN,[2*x_axis,2*y_axis]);
name_labelCN = strcat(filepath,'labelCN.mat');
name_label3D = strcat(filepath,'label_3D.mat');
data = load(name_label3D);
label3D =data.label_3D;
data = load(name_labelCN);
labelCN = data.labelCN;
label3D = label3D/max(label3D(:));
labelCN = labelCN/max(labelCN(:));

RLNE = norm(res_mean(:)-label3D(:))/norm(label3D(:))

level = 15;
figure,contour(labelCN,level),title('label');
figure,contour(resCN,level),title('JOSR-Net');

