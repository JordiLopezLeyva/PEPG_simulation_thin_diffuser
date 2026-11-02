% PEPG wavefront shaping through a thin random phase screen.
% Requires PEPG_opt_perturbed_GPU.m and Parallel Computing Toolbox.
clear; close all; clc;

%% Parameters
% The SLM sets the phase; the camera samples its Fourier plane.
Nx = 200; Ny = 200;               % SLM pixels
Nx_cam = 160; Ny_cam = 160;       % Camera pixels

% E(r) = exp(-r^2/w0^2), so the intensity falls to 1/e^2 at r = w0.
w0 = 4e-3;                        % Gaussian beam radius (1/e^2 intensity)
Lx = 3*w0; Ly = 3*w0;             % Simulated SLM dimensions [m]

% Approximate speckle width on the sampled camera grid, in pixels.
speckle_px = 5; % Size of speckles we want in unit of px
MMF_radius = 80;                 % Circular detector mask radius [camera px]

% We use one random thickness map but calculate its phase at each wavelength.
lambda = 851e-9;                  % Read wavelength [m]
lambda_write = 881e-9;
lambda_manual = 884e-9;
n_refrac = 1.5;
OPL_span_in_lambda = 30;         % Max optical path variation / lambda

% The diffuser adds a random local phase, with no absorption.
diffuser_strength = 2*pi;
N_diff = 1;                       % Size of a constant-phase block [SLM px]

target_type = 'gaussian';          % 'gaussian', 'checker', or 'image'
target_file = 'cat_cartoon_160x160.png';
target_FWHM = 5;
checker_tile = 20;

epochs = 3000;
pop_frac = 0.01;

% Hyperparameters
sigma_initial = 0.05;
learning_rate = 0.05;
sigma_alpha = 1;


plot_every = 10;
save_pop_history = true;

% This only switches between fixed phase screens; it does not write or erase material.
alternate_medium = false;         % Alternate manual/write wavelength
switch_every = 200;

seed_diffuser = 5;
seed_phase = 1234;
seed_optimizer = 12345;

%% Beam and random medium
% Physical coordinates on the SLM, centered at zero.
x = ((0:Nx-1) - (Nx-1)/2)*(Lx/Nx);
y = ((0:Ny-1) - (Ny-1)/2)*(Ly/Ny);
[X,Y] = meshgrid(x,y);

% Gaussian input field (amplitude, not intensity).
E0 = exp(-(X.^2 + Y.^2)/w0^2);

rs_diff = RandStream('Threefry','Seed',seed_diffuser);
rs_phase = RandStream('Threefry','Seed',seed_phase);

% Random phase at each pixel. N_diff > 1 gives larger constant-phase blocks.
diffuser_low = exp(1i*diffuser_strength*rand(rs_diff,round(Ny/N_diff),round(Nx/N_diff)));
rows = min(floor((0:Ny-1)*size(diffuser_low,1)/Ny)+1,size(diffuser_low,1));
cols = min(floor((0:Nx-1)*size(diffuser_low,2)/Nx)+1,size(diffuser_low,2));
diffuser = diffuser_low(rows,cols);

% Uniform random thickness: 0 to OPL_span_in_lambda*lambda/n.
% Since the thickness stays fixed, the three wavelength phases are correlated.
DeltaL_thick = (OPL_span_in_lambda*lambda/n_refrac)*rand(rs_phase,Ny,Nx);

% Local optical phase = 2*pi*n*thickness/lambda.
% This assumes the optical path is n*DeltaL, without material dispersion.
phase_read = exp(1i*2*pi*n_refrac*DeltaL_thick/lambda);
phase_write = exp(1i*2*pi*n_refrac*DeltaL_thick/lambda_write);
phase_manual = exp(1i*2*pi*n_refrac*DeltaL_thick/lambda_manual);

% A thin phase screen acts locally: E_out(x,y) = t(x,y)*E_in(x,y).
% Hence .* is equivalent to diag(t(:))*E_in(:), without storing a huge matrix.
% It is elementwise multiplication, not a dot product. The FFT mixes pixels later.
BaseField = E0 .* diffuser .* phase_read;
Write_random_matrix = diffuser .* phase_write;
manual_random_matrix = diffuser .* phase_manual;

% These phase screens share one plane; there is no propagation between them
% or simulation of a changing material.

%% Fourier plane and camera mask
% At the focal plane of a lens, the field is the Fourier transform of the SLM field.
% Roughly, speckle_px = Nfft*(Lx/Nx)/(2*w0). Here Nfft is about 672.
% Zero-padding samples the same speckle pattern more finely; it does not add modes.
Nfft_x = fftSize(2*w0*speckle_px/(Lx/Nx),Nx,Nx_cam);
Nfft_y = fftSize(2*w0*speckle_px/(Ly/Ny),Ny,Ny_cam);

% Keep the central camera region and a circular detector aperture.
[xc,yc] = meshgrid(1:Nx_cam,1:Ny_cam);
Mask = (xc-(Nx_cam+1)/2).^2 + (yc-(Ny_cam+1)/2).^2 <= MMF_radius^2;
mask_lin = find(Mask);

% Select the central Fourier frequencies using indices, avoiding GPU fftshift.
col_order = fftshift(1:Nfft_x);
row_order = fftshift(1:Nfft_y);
cam_cols = col_order((Nfft_x-Nx_cam)/2 + (1:Nx_cam));
cam_rows = row_order((Nfft_y-Ny_cam)/2 + (1:Ny_cam));
% Convert camera-mask pixels to FFT indices. Row/column vectors matter here:
% otherwise MATLAB can accidentally expand this into a huge index matrix.
[mask_rows,mask_cols] = ind2sub([Ny_cam,Nx_cam],mask_lin);
cam_rows = cam_rows(:);  % Keep both index terms as column vectors
mask_fft_lin = cam_rows(mask_rows) + (mask_cols-1)*Nfft_y;
assert(iscolumn(mask_fft_lin) && numel(mask_fft_lin)==nnz(Mask));

%% Target
% The target is a camera intensity pattern, not an optical field.
switch target_type
    case 'gaussian'
        sigma_target = target_FWHM/(2*sqrt(2*log(2)));
        r2 = (xc-(Nx_cam+1)/2).^2 + (yc-(Ny_cam+1)/2).^2;
        y_targ = exp(-r2/(2*sigma_target^2));
    case 'checker'
        y_targ = mod(floor((xc-1)/checker_tile) + floor((yc-1)/checker_tile),2);
    case 'image'
        img = double(imread(target_file));
        if size(img,3)==3
            img = 0.299*img(:,:,1) + 0.587*img(:,:,2) + 0.114*img(:,:,3);
        end
        [xi,yi] = meshgrid(linspace(1,size(img,2),Nx_cam), ...
                           linspace(1,size(img,1),Ny_cam));
        y_targ = interp2(img,xi,yi,'linear');
    otherwise
        error('Unknown target type.');
end

assert(all(isfinite(y_targ(:))) && sum(y_targ(mask_lin))>0, ...
    'Target must contain light inside the camera mask.');
% Normalize inside the mask so the loss compares intensity shapes, not power.
y_targ = y_targ/sum(y_targ(:));
target = y_targ(mask_lin);
target = target/sum(target);      % Only power inside the camera mask matters

targ_masked_2D = zeros(Ny_cam,Nx_cam);
targ_masked_2D(mask_lin) = target;

%% GPU and optimizer
% The optimization dimension is the number of SLM pixels (200*200).
g = gpuDevice();
N_dim = Nx*Ny;
pop_size = max(4,2*ceil(pop_frac*N_dim/2));

% Limit the number of simultaneous candidates to fit GPU memory.
bytes_per_candidate = 2.2*(20*N_dim + 8*Ny*Nfft_x + 8*Ny*Nx_cam ...
                         + 8*Nfft_y*Nx_cam + 12*numel(mask_lin));
batch_size = min(pop_size,max(1,floor(0.70*double(g.AvailableMemory)/bytes_per_candidate)));

% Precompute the fixed fields, leaving only the SLM phase to vary.
field_write_g = gpuArray(single(BaseField .* Write_random_matrix));
field_manual_g = gpuArray(single(BaseField .* manual_random_matrix));
target_g = gpuArray(single(target(:)));
phase_factor = complex(single(0),single(2*pi));

% PEPG uses normalized phase values q in [0,1], mapped to 2*pi*q.
rng(seed_optimizer);
gpurng(seed_optimizer);
PEPG = PEPG_opt_perturbed_GPU(N_dim,pop_size,learning_rate,learning_rate, ...
    learning_rate,N_dim/2,N_dim/2,ones(1,N_dim),sigma_initial*ones(1,N_dim));
PEPG.sigma_alpha = sigma_alpha;

mse_vals_g = gpuArray.zeros(pop_size,1,'single');
mse_best_epoch = nan(1,epochs);
sigma_mean = nan(1,epochs);
sigma_std = nan(1,epochs);
if save_pop_history
    mse_pop_history = nan(pop_size,epochs,'single');
else
    mse_pop_history = [];
end

fprintf('GPU: %s | FFT: %d x %d | population: %d | batch: %d\n', ...
    g.Name,Nfft_x,Nfft_y,pop_size,batch_size);

%% Plots
% Show the target, measured intensity and current SLM phase during training.
figImgs = figure('Name','PEPG images');
tiledlayout(1,4,'Padding','compact','TileSpacing','compact');
nexttile; imagesc(Mask); axis image off; title('Mask'); colormap(gca,'gray');
nexttile; imagesc(targ_masked_2D); axis image off; title('Target'); colormap(gca,'gray');
axC = nexttile;
imC = imagesc(zeros(Ny_cam,Nx_cam));
axis image off; title('Camera'); colormap(gca,'gray');
axP = nexttile;
imP = imagesc(zeros(Ny,Nx),[0 2*pi]);
axis image off; title('SLM phase'); colormap(gca,'hsv');

figMSE = figure('Name','PEPG loss');
phMSE = plot(nan,nan,'LineWidth',1.5);
set(gca,'XScale','log','YScale','log');
xlim([1 max(2,epochs)]); grid on;
xlabel('Epoch'); ylabel('Best masked MSE');

%% Training
best_intensity_masked = zeros(Ny_cam,Nx_cam);
best_phase = zeros(Ny,Nx);
run_tic = tic;

% PEPG proposes phase patterns; the simulated camera gives their losses.
for epoch = 1:epochs
    [mu_g,eps_g] = PEPG.ask();

    if alternate_medium && mod(floor((epoch-1)/switch_every),2)==0
        field_g = field_manual_g;
    else
        field_g = field_write_g;
    end

    % Propagate several candidate fields at once on the GPU.
    for first = 1:batch_size:pop_size
        last = min(first+batch_size-1,pop_size);
        % Each phase map is local modulation exp(i*phi) of the input field.
        q = min(max(eps_g(first:last,:) + mu_g,0),1);
        phi = permute(reshape(q,last-first+1,Ny,Nx),[2 3 1]);
        field = field_g .* exp(phase_factor*phi);

        % Coherent fields interfere in the Fourier plane before the camera measures |E|^2.
        I = cameraIntensity(field,Nfft_y,Nfft_x,cam_cols,mask_fft_lin);

        % Compare normalized masked intensity against the target.
        I = I ./ max(sum(I,1),eps('single'));
        mse_vals_g(first:last) = sum((I-target_g).^2,1).';
    end

    % Find the best candidate from this epoch before updating PEPG.
    [best_mse_g,best_idx_g] = min(mse_vals_g);
    update_plot = epoch==1 || mod(epoch,plot_every)==0 || epoch==epochs;

    if update_plot
        idx = gather(best_idx_g);
        q_best = min(max(eps_g(idx,:) + mu_g,0),1);
        phi_best = reshape(q_best,Ny,Nx);
        field_best = field_g .* exp(phase_factor*phi_best);
        I_best_g = cameraIntensity(field_best,Nfft_y,Nfft_x,cam_cols,mask_fft_lin);
        I_best_g = I_best_g/max(sum(I_best_g),eps('single'));

        best_intensity_masked(:) = 0;
        best_intensity_masked(mask_lin) = gather(I_best_g);
        best_phase = 2*pi*gather(phi_best);

        if isgraphics(imC)
            set(imC,'CData',best_intensity_masked);
            title(axC,sprintf('Camera, epoch %d',epoch));
            values = sort(best_intensity_masked(mask_lin));
            limits = values(round([0.01 0.99]*numel(values)));
            if limits(2)<=limits(1), limits = [0 max(values)]; end
            set(axC,'CLim',limits);
        end
        if isgraphics(imP)
            set(imP,'CData',best_phase);
        end
    end

    % Update the search distribution; the simulated optical medium stays fixed.
    PEPG.tell(mse_vals_g);
    diagnostics = gather([best_mse_g; mean(PEPG.sigma(:)); std(PEPG.sigma(:),1)]);
    mse_best_epoch(epoch) = diagnostics(1);
    sigma_mean(epoch) = diagnostics(2);
    sigma_std(epoch) = diagnostics(3);

    if save_pop_history
        mse_pop_history(:,epoch) = gather(mse_vals_g);
    end
    if update_plot
        if isgraphics(phMSE)
            set(phMSE,'XData',1:epoch,'YData',max(mse_best_epoch(1:epoch),1e-16));
        end
        drawnow;
    end
    if epoch==1 || mod(epoch,200)==0 || epoch==epochs
        fprintf('Epoch %d/%d | MSE %.3e | sigma %.3e\n', ...
            epoch,epochs,mse_best_epoch(epoch),sigma_mean(epoch));
    end
end
run_seconds = toc(run_tic);

%% Save
% Save the final epoch's best displayed pattern and training history.
outdir = fullfile(pwd,['pepg_' datestr(now,'yyyymmdd_HHMMSS')]);
mkdir(outdir);
I_best = best_intensity_masked;
imwrite(I_best/max(I_best(:)),fullfile(outdir,'IntensityMasked.png'));
hue = mod(best_phase/(2*pi),1);
imwrite(hsv2rgb(cat(3,hue,ones(size(hue)),ones(size(hue)))), ...
    fullfile(outdir,'PhaseHue.png'));
if isgraphics(figImgs), savefig(figImgs,fullfile(outdir,'LiveImages.fig')); end
if isgraphics(figMSE), savefig(figMSE,fullfile(outdir,'LiveMSE.fig')); end
save(fullfile(outdir,'results_single_lambda.mat'), ...
    'mse_best_epoch','mse_pop_history','sigma_mean','sigma_std','run_seconds', ...
    'best_intensity_masked','best_phase','I_best','y_targ','targ_masked_2D', ...
    'Nx','Ny','Nx_cam','Ny_cam','Nfft_x','Nfft_y','Mask','MMF_radius', ...
    'w0','Lx','Ly','speckle_px','lambda','lambda_write','lambda_manual', ...
    'n_refrac','DeltaL_thick','BaseField','Write_random_matrix','manual_random_matrix', ...
    'seed_diffuser','seed_phase','seed_optimizer', ...
    'sigma_initial','learning_rate','sigma_alpha','epochs','pop_frac','pop_size', ...
    'target_type','target_file','batch_size','-v7.3');
fprintf('Completed in %.1f s. Results saved to %s\n',run_seconds,outdir);

%% Functions
% Find a nearby FFT size with small prime factors for faster GPU transforms.
function Nfft = fftSize(target,nSlm,nCam)
    candidates = max(nSlm,nCam):max([nSlm,nCam,ceil(1.5*target)+100]);
    candidates = candidates(mod(candidates-nSlm,2)==0 & mod(candidates-nCam,2)==0);
    good = arrayfun(@(n) all(ismember(factor(n),[2 3 5 7])),candidates);
    candidates = candidates(good);
    if isempty(candidates), error('No suitable FFT size found.'); end
    [~,idx] = min(abs(candidates-target));
    Nfft = candidates(idx);
end

% The 2D Fourier transform is done along x and then y.
% Crop x-frequencies first to save memory, then keep only masked camera pixels.
% Camera intensity is abs(F).^2; propagation prefactors cancel on normalization.
function I = cameraIntensity(field,Nfft_y,Nfft_x,cam_cols,mask_fft_lin)
    nCandidates = size(field,3);
    F = fft(field,Nfft_x,2);
    F = fft(F(:,cam_cols,:),Nfft_y,1);
    F = reshape(F,Nfft_y*numel(cam_cols),nCandidates);
    F = F(mask_fft_lin,:);
    I = abs(F).^2;
end


PEPG_simulation_of_thin_diffuser
