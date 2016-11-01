P_streched = 255/191*imsubtract(P,13);
imshow(P_streched)
P_streched = 255/191*(P-13);

h1 = fspecial('gaussian',[5 5],1);
h2 = fspecial('gaussian',[5 5],2);
P_double = im2double(P);
P1_double = conv2(P_double,h1);
imshow(P1_double)
P2_double = conv2(P_double,h2);
imshow(P2_double)
P1 = imfilter(P,h1);
P2 = imfilter(P,h2);
P = imread('resource/ntusp.jpg');
P1 = imfilter(P,h1);
P2 = imfilter(P,h2);

P1 = medfilt2(P);
P2 = medfilt2(P, [5 5]);

F = fft2(P);
S=abs(F).^2;
imagesc(fftshift(S.^0.1));
F(15:19, 247:251) = 0;
P2=ifft2(F);
P2=real(P2);
P2=P2+130;
P2=uint8(P2);
imshow(P2)
for i = 1:256
    for j = 1:256
        if S(i,j)> 3*10^9
            for u=-2:2
                for v = -2:2
                    if i+u>0 && i+u<257 && j+v>0 && j+v<257
                        F(i,j)=0;
                    end
                end
            end
        end
    end
end

v=[0;0;210;0;210;290;0;290]
A = [
    X(1),Y(1),1,0,0,0,-x(1)*X(1), -x(1)*Y(1);
    0,0,0,X(1),Y(1),1,-y(1)*X(1), -y(1)*Y(1);
    X(2),Y(2),1,0,0,0,-x(2)*X(2), -x(2)*Y(2);
    0,0,0,X(2),Y(2),1,-y(2)*X(2), -y(2)*Y(2);
    X(3),Y(3),1,0,0,0,-x(3)*X(3), -x(3)*Y(3);
    0,0,0,X(3),Y(3),1,-y(3)*X(3), -y(3)*Y(3);
    X(4),Y(4),1,0,0,0,-x(4)*X(4), -x(4)*Y(4);
    0,0,0,X(4),Y(4),1,-y(4)*X(4), -y(4)*Y(4);
]
