function map = disparity_map(Pl, Pr)
[height, width] = size(Pl);
map = ones(height-10, width-10);
for row = 6:height-5
    for xl = 6:width-5
        T = Pl(row-5:row+5,xl-5:xl+5);
        left = xl-14;
        right = xl;
        if left<6
            left = 6;
        end
        ssd_min = Inf;
        xr_min = left;
        for xr = left:right
            I = Pr(row-5:row+5,xr-5:xr+5);
            I_flipped = rot90(I,2);
            ssd_1 = ifft2(fft2(I).*fft2(I_flipped));
            ssd_1 = ssd_1(11,11);
            ssd_2 = ifft2(fft2(T).*fft2(I_flipped));
            ssd_2 = ssd_2(11,11)*2;
            ssd = ssd_1 - ssd_2;
            if ssd<ssd_min
                ssd_min=ssd;
                xr_min = xr;
            end
        end
        d = xl - xr_min;
        map(row-5, xl-5) = d;
    end
end