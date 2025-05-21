filename = '../test/EM_T/test128.mat';
filepath = '../test/EM_T/EM/5e-3/';
data = load(filename);
data = data.FID1;
label_f = fft2(data);
[M,N]=size(label_f);
label_f = label_f/max(real(label_f(:))); %Pure label spectrum
label_t = ifft2(label_f);

load(strcat(filepath,'factor.mat'));
load(strcat(filepath,'rec_1FID.mat'));
rec1_real=reshape(rec1(:,1,:,1),[M,N]);
rec1_imag=reshape(rec1(:,1,:,2),[M,N]);
rec1_complex=rec1_real+1i*rec1_imag;
rec1_out = factor'.*rec1_complex;
max_rec1 = max(real(rec1_out(:)));
res1=rec1_out/max_rec1;

RLNE = norm(real(res1(:))-real(label_f(:)))/norm(real(label_f(:)))