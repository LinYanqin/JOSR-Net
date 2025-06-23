function [factor,M] = save2DUnderSampledSpectrumToTXT( FID,peaknumber,iter)
F=fftshift(fft2(FID));
temp = real(F(:));
factor = max(temp(:));
F=F/factor;
Size=size(F);
N1=Size(1);
N2=Size(2);
M=zeros(N1,N2,2);
M(:,:,1)=real(F);
M(:,:,2)=imag(F);
M=single(M);
% datapath='../../DPSdata/20241021_1130/Input/';
% FileName=['2D_frequencydomain_',num2str(peaknumber),'peaks_',num2str(iter),'_x.txt'];
% dlmwrite(strcat(datapath,FileName), M,'delimiter' , ' ', 'newline', 'unix');
% newline shi huan hang fu lei xing 'pc' he 'unix'
end
