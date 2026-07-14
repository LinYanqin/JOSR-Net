function [S] = generate1DFID_onepeak_SP(Amplitude,Omega1,Tao1,N1,fs1,offdata,enddata)  
n1=0:1:N1-1;
t1=n1/fs1;
S=Amplitude*exp(1i*2*pi*Omega1*t1).*exp(-t1/Tao1);
S=applySineWindow(S,offdata,enddata,1,0.5);
FID1fft = fft(S);
theta = auto_phase(FID1fft,1);
FID1fft = FID1fft .* exp(1j*theta);
S = ifft(FID1fft);
end
