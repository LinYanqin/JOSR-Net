function [fid] = generate2DFID_onepeak_SP(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2,offdata,enddata)
t1=0:1/fs1:(N1-1)/fs1;
t2=0:1/fs2:(N2-1)/fs2;
S1=Amplitude*exp(1i*2*pi*Omega1*t1).*exp(-t1/Tao1);
S1 = applySineWindow(S1,offdata,enddata,1,0.5);
S2 = exp(1i*2*pi*Omega2*t2).*exp(-t2/Tao2);
S2 = applySineWindow(S2,offdata,enddata,1,0.5);
% theta1 = 1.5*rand(1);
% S1 = S1*exp(1i*theta1);
% figure,plot(real(fft(S1)));
% theta2 = 1.5*rand(1);
% S2 = S2*exp(1i*theta2);
% figure,plot(real(fft(S2)));
R1 = real(S1);
I1 = imag(S1);
R2 = real(S2);
I2 = imag(S2);
fid = zeros(2*N1,2*N2);
R1R2 = R1.'*R2;
R1I2 = R1.'*I2;
I1R2 = I1.'*R2;
I1I2 = I1.'*I2;
fid(1:2:end,1:2:end) = R1R2;
fid(1:2:end,2:2:end) = R1I2;
fid(2:2:end,1:2:end) = I1R2;
fid(2:2:end,2:2:end) = I1I2;
end