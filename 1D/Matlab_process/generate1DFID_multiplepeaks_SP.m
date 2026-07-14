function [FID] = generate1DFID_multiplepeaks_SP(Amplitude,Omega1,Tao1,N1,fs1,offdata,enddata)
peak_number=length(Amplitude);
for iter=1:peak_number
    if iter==1
        FID=generate1DFID_onepeak_SP(Amplitude(iter),Omega1(iter),Tao1(iter),N1,fs1,offdata,enddata);
    else
        FID=FID+generate1DFID_onepeak_SP(Amplitude(iter),Omega1(iter),Tao1(iter),N1,fs1,offdata,enddata);
    end
end
end