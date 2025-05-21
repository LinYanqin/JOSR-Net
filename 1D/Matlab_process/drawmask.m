clear;
M=128;
filepath='../test/BMRB_data/DPSmask.mat';
load(filepath);
mask = DPSmask;
figure;
hold on;
for i=1:M
    p1 = [i,i];  %x
    p2 = [0, mask(i)];
    line(p1, p2,'Color','r','LineWidth',1.2);
end
hold off;
title('JOSR-Netmask');