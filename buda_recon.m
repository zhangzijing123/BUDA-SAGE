clear;clc
% close all;

addpath(genpath('/home/zzj/MRI/autofs/tools/'))


%--------------------------------------------------------------------------
%% fast-sms-BUDA
%--------------------------------------------------------------------------

data_path = '/data/';

load([data_path, 'kdata.mat'])       % the kspace data
              
load([data_path, 'sens_gre.mat'])    % the coil sensitivity map

load([data_path, 'b0_fieldmap.mat']) % the B0 map

load([data_path, 'esp.mat'])         % the echo espacing



AccY = 8;      % the acceleration rate of ky direction
AccZ = 2;      % the acceleration rate of kz direction

num_shot = 2;  % the number of shot of ap/pa

[N(1),N(2), num_chan, num_slc_raw] = size(kspace_cor_ap);
num_slc = num_slc_raw * AccZ;


%% 

num_shot_ap_pa = 2;
ky_idx_ap = 2 : (AccY/num_shot) : N(2);
ky_idx_pa = 4 : (AccY/num_shot) : N(2);

PE_line = length(ky_idx_ap);

t_value_ap = [0:(AccY/num_shot):(PE_line*(AccY/num_shot)-1)] * esp;
t_value_pa = t_value_ap(end:-1:1);

winSize = [1,1] * 7;
lambda_msl = 1.50;

step_size = .6;
num_iter = 200;
tol = 0.1;

keep = 1:floor(lambda_msl*prod(winSize));

img_msl = zeross([N,4,num_slc_raw]);


for ii_slc= 20%1:num_slc_raw
    slc_select = [ii_slc,ii_slc+num_slc/AccZ];
    disp(['slice: ', num2str(slc_select)])
    
    % AP data select
    kspace_ap_collaps = kspace_cor_ap(:,:,:,slc_select(1));
    kspace_ap = kspace_ap_collaps(:,ky_idx_ap,:);
        
    % PA data select
    kspace_pa_collaps = kspace_cor_pa(:,:,:,slc_select(1));
    kspace_pa = kspace_pa_collaps(:,ky_idx_pa,:);
        
    signal_ap = ifftc(kspace_ap,1);
    signal_pa = ifftc(kspace_pa,1);
    
    % sens select
    sens_gre_shift = sens_gre(:,:,:,slc_select);
    sens = zeross(size(sens_gre_shift));

    sens1 = sens_gre(:,:,:,slc_select(1));
    sens2 = sens_gre(:,:,:,slc_select(2));
    
    
    B0_select1 = 2*pi * b0_fieldmap(:,:,slc_select(1));
    B0_select2 = 2*pi * b0_fieldmap(:,:,slc_select(2));
    
    E = fftc(eye(N(2)),1);
    E_ap = E(ky_idx_ap, : );
    E_pa = E(ky_idx_pa, : );

    % create and store encoding matrix
    EWC_ap = zeross([num_chan*PE_line, N(2)*2, N(1)]);
    EWC_pa = zeross([num_chan*PE_line, N(2)*2, N(1)]);
    
    W_ap1 = exp(1i*mtimesx(repmat(t_value_ap.',[1 1 N(1)]),permute(B0_select1,[3 2 1])));
    W_ap2 = exp(1i*mtimesx(repmat(t_value_ap.',[1 1 N(1)]),permute(B0_select2,[3 2 1])));
    W_pa1 = exp(1i*mtimesx(repmat(t_value_pa.',[1 1 N(1)]),permute(B0_select1,[3 2 1])));
    W_pa2 = exp(1i*mtimesx(repmat(t_value_pa.',[1 1 N(1)]),permute(B0_select2,[3 2 1])));
    EW_ap = bsxfun(@times, repmat(E_ap,1,2),cat(2,W_ap1,W_ap2));
    EW_pa = bsxfun(@times, repmat(E_pa,1,2),cat(2,W_pa1,W_pa2));
        
    for c = 1:num_chan
        EWC_ap(1 + (c-1)*PE_line : c*PE_line, :, :) = bsxfun(@times, EW_ap, cat(2,permute(sens1(:,:,c),[3,2,1]), permute(sens2(:,:,c),[3 2 1])));
        EWC_pa(1 + (c-1)*PE_line : c*PE_line, :, :) = bsxfun(@times, EW_pa, cat(2,permute(sens1(:,:,c),[3,2,1]), permute(sens2(:,:,c),[3 2 1])));
    end
    
    EWC_apN = mtimesx(EWC_ap,'c',EWC_ap);
    EWC_paN = mtimesx(EWC_pa,'c',EWC_pa);

    sgnl_ap = permute(signal_ap, [2 3 1]);
    svec_ap = reshape(sgnl_ap, num_chan*PE_line, 1, N(1));
    EWC_apHsvec = sq(mtimesx(EWC_ap,'c',svec_ap));
    
    sgnl_pa = permute(signal_pa, [2 3 1]);
    svec_pa = reshape(sgnl_pa, num_chan*PE_line, 1, N(1));
    EWC_paHsvec = squeeze(mtimesx(EWC_pa,'c',svec_pa));

    
    temp = zeross([N(1), N(2)*2, 2]);
    im_rec = zeross([N(2)*2, 2 , N(1)]);
    im_rec_sms = zeross([N, 4]);

    for iter = 1:num_iter
            
        im_prev = im_rec;

        im_rec(:,1,:) = sq(im_rec(:,1,:))- step_size * (sq(mtimesx(EWC_apN,im_rec(:,1,:)))-EWC_apHsvec);
        im_rec(:,2,:) = sq(im_rec(:,2,:))- step_size * (sq(mtimesx(EWC_paN,im_rec(:,2,:)))-EWC_paHsvec);

        % mussels constraint
        im_rec = permute(im_rec, [3,1,2]);

        im_rec_sms(:,:,1) = im_rec(:,1:end/2,1);
        im_rec_sms(:,:,2) = im_rec(:,1:end/2,2);
        im_rec_sms(:,:,3) = im_rec(:,end/2+1:end,1);
        im_rec_sms(:,:,4) = im_rec(:,end/2+1:end,2);
        im_rec = im_rec_sms;


        % slice 1st 
        A = Im2row( fft2call(im_rec(:,:,1:2)), winSize );
           
        [U, S, V] = svdecon(A); 
        U(isnan(U))=0;S(isnan(S))=0;V(isnan(V))=0;
        U(isinf(U))=0;S(isinf(S))=0;V(isinf(V))=0;

        A = U(:,keep) * S(keep,keep) * V(:,keep)';

        k_pocs = Row2im(A, [N, num_shot_ap_pa], winSize);
        im_rec(:,:,1:2) = ifft2call(k_pocs);

        % slice 2nd
        A = Im2row( fft2call(im_rec(:,:,3:4)), winSize );
           
        [U, S, V] = svdecon(A);         
        U(isnan(U))=0;S(isnan(S))=0;V(isnan(V))=0;
        U(isinf(U))=0;S(isinf(S))=0;V(isinf(V))=0;
       
        A = U(:,keep) * S(keep,keep) * V(:,keep)';

        k_pocs = Row2im(A, [N, num_shot_ap_pa], winSize);
        im_rec(:,:,3:4) = ifft2call(k_pocs);
       
        im_rec_disp = im_rec;


        temp(:,1:end/2,1) = im_rec(:,:,1); 
        temp(:,1:end/2,2) = im_rec(:,:,2);
        temp(:,end/2+1:end,1) = im_rec(:,:,3);
        temp(:,end/2+1:end,2) = im_rec(:,:,4);
        im_rec = temp;
        im_rec = permute(im_rec, [2,3,1]);
            
        update = rmse(im_prev,im_rec);
        mosaic(im_rec_disp, 2, 2, 1, ['iter: ', num2str(iter), '  update: ', num2str(rmse(im_prev,im_rec))], [0,.4e-3], 90)
        if update < tol
            break
        end
    end
    
    img_msl(:,:,:,ii_slc) = im_rec_disp;
        
end


img_msl = reshape(permute(reshape(img_msl,[N(1),N(2),2,2,num_slc_raw]),[1,2,3,5,4]),[N(1),N(2),2,num_slc]);

mosaic(mean(abs(img_msl(:,:,:,slc_select)),3), 1, 2, 11, 'buda', [0,.4e-3], 90)
% mosaic(mean(abs(img_msl),3), 6, 10, 101, 'buda', [0,.4e-3], 90)






