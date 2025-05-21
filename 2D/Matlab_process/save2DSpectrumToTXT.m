function [M] = save2DSpectrumToTXT(FID,peaknumber,iter,f)
Size=size(FID);
N1=Size(1);
N2=Size(2);
F=fftshift(fft2(FID));
F=F/f;
M=zeros(N1,N2,2);
M(:,:,1)=real(F);
M(:,:,2)=imag(F);
M=single(M);
% datapath='../../DPSdata/20241021_1130/Label/';
% FileName=['2D_frequencydomain_',num2str(peaknumber),'peaks_',num2str(iter),'_y.txt'];
% dlmwrite(strcat(datapath,FileName), M,'delimiter' , ' ', 'newline', 'unix');
end
