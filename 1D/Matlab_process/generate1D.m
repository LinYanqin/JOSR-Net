N1=128;
N2=128;
c = 1;
max_peak = 10;
savepath = '../JOSRdata/';
maskpath = '../JOSRdata/mask_pos128_32.mat';
load(maskpath);
for p = 1:max_peak
    for iter=1:2000
        input_data = zeros(1,N2,2);
        label_data = zeros(1,N2,2);
        mask_index = mask_pos(c,:);
        mask = zeros(1,N2);
        l = length(mask_index);
        for k = 1:1:l
            mask(1,mask_index(1,k)+1) = 1;
        end
        FID1 = 0;
        fs = 4000 + (1300-1000)*rand(1);
        for i = 1:p
            Amplitude=0.9*rand(1); 
            Omega = (fs-300)*rand(1);
            Tao1=0.02+(0.01)*rand(1);  
            FID1 = FID1 + generate1DFID_multiplepeaks(Amplitude,Omega,Tao1,N1,fs); 
        end
        % zerofill
        FID11=zeros(1,N2);
        FID11(1,1:N1)=FID1;

        FID2=FID11.*mask;
        [f,input] = saveUnderSampledSpectrumToTXT(FID2,p,iter);
        label = saveSpectrumToTXT(FID11,p,iter,f);
        
        input_data(1,:,1) = input(:,1);
        input_data(1,:,2) = input(:,2);
        label_data(1,:,1) = label(:,1);
        label_data(1,:,2) = label(:,2); 

        datapath=strcat(savepath,'Label/');
        FileName=['1D_frequencydomain_',num2str(p),'peaks_',num2str(iter),'_y.txt'];
        dlmwrite(strcat(datapath,FileName), label_data,'delimiter' , ' ', 'newline', 'unix');

        c = c + 1
    end
end