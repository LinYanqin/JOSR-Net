clear;
DPS_rows = 32;
DPS_cols = 64;
filepath = '../Processed_data/BMRB_data/DPSmask.mat';
load(filepath);
figure,spy(DPSmask(1:DPS_rows,1:DPS_cols));title('DPSmask');
