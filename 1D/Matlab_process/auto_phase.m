function theta=auto_phase(spec,mode)
    N1=128;
    N2=100;
    np=length(spec);
    temp=zeros(1,100);
    dw=2*pi/10000;
    if mode==1
        for k=1:N1
            theta=2*pi/N1*(k-1);
            spec_temp=spec*exp(1i*theta);
            spec_temp=real(spec_temp);
%             temp(k)=sum(spec_temp);
            temp(k)=sum( (spec_temp(spec_temp>0)).^2 )/sum( (spec_temp(spec_temp<0)).^2 );
        end
        [maxi kk]=max(temp);
        theta=2*pi/N1*(kk-1);
    else if mode==2
        for k=1:N1
            for n=1:N2
                theta=2*pi/N1*(k-1)+[0:(np-1)]*dw*(n-1);
                spec_temp=spec.*exp(1i*theta);
                spec_temp=real(spec_temp);
                temp(k,n)=sum(spec_temp);
            end
        end
        [maxi kk]=max(temp);
        [maxii kn]=max(maxi);
        theta=2*pi/N1*(kk(kn)-1)+[0:(np-1)]*dw*(kn-1);
    end
    end
end