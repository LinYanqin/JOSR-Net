N1 = 32;
N2 = 64;
c = 1;
max_peak = 20;
savepath = '../JOSRdata/';
maskpath = '../JOSRdata/mask_pos32_64_123.mat';
load(maskpath);
for p = 1:max_peak
    for iter=1:1000
        FID1=0;
        rand_off = randperm(4,1);
        rand_end = randperm(4,1);
        off_data = [0.3,0.4,0.5,0.6];
        enddata = [0.85,0.9,0.95,1];
        for i = 1:p
            Amplitude=0.01+(1-0.01)*rand(1);
            Tao1=0.01+(0.01)*rand(1);
            Tao2=0.01+(0.01)*rand(1);
            fs1=4000; 
            Omega1=fs1*(0.05 + (0.95-0.05)*rand(1));
            fs2=4000;
            Omega2=fs2*(0.05 + (0.95-0.05)*rand(1));
            FID1=FID1+generate2DFID_multiplepeaks_SP(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2,off_data(rand_off),enddata(rand_end));
        end
        R1R2 = FID1(1:2:end,1:2:end,:);
        R1I2 = FID1(1:2:end,2:2:end,:);
        I1R2 = FID1(2:2:end,1:2:end,:);
        I1I2 = FID1(2:2:end,2:2:end,:);
        FID_real1_down = R1R2+1i*R1I2;
        FID_real2_down = I1R2+1i*I1I2; %Recombine data
        % zero fill
        FID_real1 = zeros(64,64);
        FID_real2 = zeros(64,64);
        FID_real1(1:N1,1:N2)=FID_real1_down;
        FID_real2(1:N1,1:N2)=FID_real2_down;
        index = permute(mask_pos((c+1)/2,:,:),[2,3,1]);
        num = length(index);
        mask= zeros(64,64);
        for i = 1:num
            mask((index(i,1)+1),(index(i,2)+1)) = 1; %Generate mask
        end
        FID_real1_under = FID_real1.*mask;
        FID_real2_under = FID_real2.*mask;
        [f1,input_real1] = save2DUnderSampledSpectrumToTXT(FID_real1_under,p,2*iter-1);
        [f2,input_real2] = save2DUnderSampledSpectrumToTXT(FID_real2_under,p,2*iter);
        
        input = zeros(64, 64, 4);
        input(:,:,1:2) = input_real1;
        input(:,:,3:4) = input_real2;
        
        label1 = save2DSpectrumToTXT(FID_real1,p,2*iter-1,f1);
        label2 = save2DSpectrumToTXT(FID_real2,p,2*iter,f2);
        
        label = zeros(64, 64, 4);
        label(:,:,1:2) = label1;
        label(:,:,3:4) = label2;
        
        mask = single(mask);
        
        datapath=strcat(savepath,'Label/');
        FileName=['2D_frequencydomain_',num2str(p),'peaks_',num2str(2*iter),'_y.txt'];
        dlmwrite(strcat(datapath,FileName), label,'delimiter' , ' ', 'newline', 'unix');
        
        
        c = c + 2
    end
end