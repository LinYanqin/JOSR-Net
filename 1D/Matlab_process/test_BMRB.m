filename = '../test/BMRB_data/';
filepath = '../test/BMRB_data/';
label = load(strcat(filename,'test.mat'));
label = label.data;
label = fliplr(label);
[M,N]=size(label);
label_f = label/max(real(label(:)));
label_t = ifft2(label_f);

load(strcat(filepath,'factor.mat'));
load(strcat(filepath,'rec_1FID.mat'));
rec1_real=reshape(rec1(:,1,:,1),[M,N]);
rec1_imag=reshape(rec1(:,1,:,2),[M,N]);
rec1_complex=rec1_real+1i*rec1_imag;
rec1_out = factor'.*rec1_complex; 
max_rec1 = max(real(rec1_out(:)));
res1=rec1_out/max_rec1;

level = 10;
figure,contour(abs(label_f),level);title('label')
figure,contour(abs(res1),level);title('JOSR-Net')
RLNE_JOSR = norm(real(res1(:))-real(label_f(:)))/norm(real(label_f(:)))