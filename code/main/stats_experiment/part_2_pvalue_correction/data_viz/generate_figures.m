%% Plot distributions with parameter variation and generate parameter tables

% Load data file containing parameters
data_file = 'C:\Users\CDoga\Documents\Research\EEG-Neuroscience-PhD\code\main\stats_experiment\part_2_pvalue_correction\sign_swap_40k_perms_n40_2026-01-03_14-54-10.parquet';
data = parquetread(data_file);

% Compute parameter statistics directly from Param_* columns
param_stats = computeParameterStatsFromTable(data);

% Generate x-axis range
x = linspace(-3, 3, 1000);

% Extract parameter statistics for NON-SYMMETRIC distributions only
gamma_stats = param_stats(strcmp({param_stats.distribution}, 'GAMMA'));
exp_stats = param_stats(strcmp({param_stats.distribution}, 'EXPONENTIAL'));
lognorm_stats = param_stats(strcmp({param_stats.distribution}, 'LOGNORMAL'));

%% ==========================
% GAMMA distributions
% ==========================

n_samples = 300;

gamma_low  = [];   % shape < 1
gamma_high = [];   % shape >= 1

z = linspace(0,10,2000);   % natural Gamma domain

for i = 1:n_samples
    
    % ---- sample from empirical parameter distribution ----
    shape = gamma_stats.mean.shape + gamma_stats.std.shape * randn();
    shape = min(max(shape, gamma_stats.min.shape), gamma_stats.max.shape);

    scale = gamma_stats.mean.scale + gamma_stats.std.scale * randn();
    scale = min(max(scale, gamma_stats.min.scale), gamma_stats.max.scale);

    mu = shape * scale;

    % ---- compute pdf on natural domain ----
    pdf_z = gampdf(z, shape, scale);

    % ---- shift mean to 0 ----
    x_shift = z - mu;

    % ---- interpolate onto common x grid ----
    pdf_x = interp1(x_shift, pdf_z, x, 'linear', 0);

    % ---- split by regime ----
    if shape < 1
        gamma_low(end+1,:) = pdf_x;
    else
        gamma_high(end+1,:) = pdf_x;
    end
end

% ===== summarise =====
gamma_low_mean  = mean(gamma_low,1);
gamma_low_lo    = prctile(gamma_low,5,1);
gamma_low_hi    = prctile(gamma_low,95,1);

gamma_high_mean = mean(gamma_high,1);
gamma_high_lo   = prctile(gamma_high,5,1);
gamma_high_hi   = prctile(gamma_high,95,1);

%% ==========================
% EXPONENTIAL distributions
% ==========================

n_samples = 100;
exp_pdfs = zeros(n_samples, length(x));

for i = 1:n_samples
    rate = exp_stats.mean.rate + exp_stats.std.rate * randn();
    rate = max(0.3, min(2.5, rate));
    
    exp_mean = 1/rate;
    pdf_temp = exppdf(x + exp_mean, 1/rate);
    pdf_temp(x < -exp_mean) = 0;
    exp_pdfs(i,:) = pdf_temp;
end

exp_pdf_mean  = mean(exp_pdfs,1);
exp_pdf_lower = prctile(exp_pdfs,5,1);
exp_pdf_upper = prctile(exp_pdfs,95,1);

%% ==========================
% LOGNORMAL distributions
% ==========================

lognorm_pdfs = zeros(n_samples, length(x));

for i = 1:n_samples
    mu = lognorm_stats.mean.mu + lognorm_stats.std.mu * randn();
    mu = max(lognorm_stats.min.mu, min(lognorm_stats.max.mu, mu));

    sigma = lognorm_stats.mean.sigma + lognorm_stats.std.sigma * randn();
    sigma = max(lognorm_stats.min.sigma, min(lognorm_stats.max.sigma, sigma));
    
    lognorm_mean = exp(mu + sigma^2/2);
    pdf_temp = lognpdf(x + lognorm_mean, mu, sigma);
    pdf_temp(x < -lognorm_mean) = 0;
    lognorm_pdfs(i,:) = pdf_temp;
end

lognorm_pdf_mean  = mean(lognorm_pdfs,1);
lognorm_pdf_lower = prctile(lognorm_pdfs,5,1);
lognorm_pdf_upper = prctile(lognorm_pdfs,95,1);

%% ==========================
% SYMMETRIC distributions
% ==========================

pdf_normal  = normpdf(x,0,1);

pdf_uniform = zeros(size(x));
in_range = (x >= -0.5) & (x <= 0.5);
pdf_uniform(in_range) = 1/(0.5-(-0.5));

pdf_studentt = tpdf(x,3);

pdf_laplace = 0.5*exp(-abs(x));

%% ==========================
% FIGURE
% ==========================

figure('Position',[100,100,1600,800]);

%% Row 1 — NON-SYMMETRIC

% --- Gamma shape < 1 ---
subplot(2,4,1);
hold on;
fill([x fliplr(x)], [gamma_low_lo fliplr(gamma_low_hi)], ...
     [1 0.6 0.6], 'FaceAlpha',0.3,'EdgeColor','none');
plot(x, gamma_low_mean,'r-','LineWidth',2.5);
xline(0,'k--','LineWidth',1.5);
hold off;
xlabel('x'); ylabel('Probability Density');
title('Gamma (shape < 1)');
legend('90% CI','Mean PDF');
grid on; xlim([-4 4]); set(gca,'FontSize',12);


% --- Gamma shape ≥ 1 ---
subplot(2,4,2);
hold on;
fill([x fliplr(x)], [gamma_high_lo fliplr(gamma_high_hi)], ...
     [0.7 0.85 1], 'FaceAlpha',0.3,'EdgeColor','none');
plot(x, gamma_high_mean,'b-','LineWidth',2.5);
xline(0,'k--','LineWidth',1.5);
hold off;
xlabel('x'); ylabel('Probability Density');
title('Gamma (shape ≥ 1)');
legend('90% CI','Mean PDF');
grid on; xlim([-4 4]); set(gca,'FontSize',12);


% --- Exponential ---
subplot(2,4,3);
hold on;
fill([x fliplr(x)], [exp_pdf_lower fliplr(exp_pdf_upper)], ...
     [0.7 1 0.7],'FaceAlpha',0.3,'EdgeColor','none');
plot(x, exp_pdf_mean,'g-','LineWidth',2.5);
xline(0,'k--','LineWidth',1.5);
hold off;
xlabel('x'); ylabel('Probability Density');
title('Exponential');
grid on; xlim([-4 4]); set(gca,'FontSize',12);


% --- Lognormal ---
subplot(2,4,4);
hold on;
fill([x fliplr(x)], [lognorm_pdf_lower fliplr(lognorm_pdf_upper)], ...
     [0.7 0.7 1],'FaceAlpha',0.3,'EdgeColor','none');
plot(x, lognorm_pdf_mean,'b-','LineWidth',2.5);
xline(0,'k--','LineWidth',1.5);
hold off;
xlabel('x'); ylabel('Probability Density');
title('Lognormal');
grid on; xlim([-4 4]); set(gca,'FontSize',12);

%% Row 2 — SYMMETRIC

subplot(2,4,5);
plot(x, pdf_normal,'LineWidth',2.5);
xline(0,'k--','LineWidth',1.5);
title('Normal'); grid on; xlim([-2 2]);

subplot(2,4,6);
plot(x, pdf_uniform,'LineWidth',2.5);
xline(0,'k--','LineWidth',1.5);
title('Uniform'); grid on; xlim([-2 2]);

subplot(2,4,7);
plot(x, pdf_studentt,'LineWidth',2.5);
xline(0,'k--','LineWidth',1.5);
title('Student-t'); grid on; xlim([-2 2]);

subplot(2,4,8);
plot(x, pdf_laplace,'LineWidth',2.5);
xline(0,'k--','LineWidth',1.5);
title('Laplace'); grid on; xlim([-2 2]);

sgtitle('All Distributions: Non-Symmetric (with variation) vs Symmetric (fixed)','FontSize',16,'FontWeight','bold');


% Print comprehensive parameter table
fprintf('\n');
fprintf('========================================================================\n');
fprintf('                   PARAMETER STATISTICS TABLE\n');
fprintf('========================================================================\n');
fprintf('\n');

% NON-SYMMETRIC DISTRIBUTIONS
fprintf('NON-SYMMETRIC DISTRIBUTIONS (WITH PARAMETER VARIATION):\n');
fprintf('========================================================================\n');

% GAMMA Distribution
fprintf('\nGAMMA DISTRIBUTION:\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('Parameter    Mean        Std Dev     Min         Max\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('shape        %.4f      %.4f      %.4f      %.4f\n', ...
    gamma_stats.mean.shape, gamma_stats.std.shape, gamma_stats.min.shape, gamma_stats.max.shape);
fprintf('scale        %.4f      %.4f      %.4f      %.4f\n', ...
    gamma_stats.mean.scale, gamma_stats.std.scale, gamma_stats.min.scale, gamma_stats.max.scale);
fprintf('------------------------------------------------------------------------\n');

% EXPONENTIAL Distribution
fprintf('\nEXPONENTIAL DISTRIBUTION:\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('Parameter    Mean        Std Dev     Min         Max\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('rate         %.4f      %.4f      %.4f      %.4f\n', ...
    exp_stats.mean.rate, exp_stats.std.rate, exp_stats.min.rate, exp_stats.max.rate);
fprintf('------------------------------------------------------------------------\n');

% LOGNORMAL Distribution
fprintf('\nLOGNORMAL DISTRIBUTION:\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('Parameter    Mean        Std Dev     Min         Max\n');
fprintf('------------------------------------------------------------------------\n');
fprintf('mu           %.4f      %.4f      %.4f      %.4f\n', ...
    lognorm_stats.mean.mu, lognorm_stats.std.mu, lognorm_stats.min.mu, lognorm_stats.max.mu);
fprintf('sigma        %.4f      %.4f      %.4f      %.4f\n', ...
    lognorm_stats.mean.sigma, lognorm_stats.std.sigma, lognorm_stats.min.sigma, lognorm_stats.max.sigma);
fprintf('------------------------------------------------------------------------\n');

% SYMMETRIC DISTRIBUTIONS
fprintf('\n\nSYMMETRIC DISTRIBUTIONS (FIXED PARAMETERS):\n');
fprintf('========================================================================\n');
fprintf('These distributions use fixed theoretical parameters with no variation.\n');
fprintf('------------------------------------------------------------------------\n');

fprintf('\n');
fprintf('========================================================================\n');
    
% Save figure
saveas(gcf, 'C:\Users\CDoga\Documents\Research\EEG-Neuroscience-PhD\code\main\stats_experiment\part_2_pvalue_correction\data_viz\plots\all_distributions_comparison.png');

%% Helper function to compute parameter statistics from Param_* columns in table
function param_stats = computeParameterStatsFromTable(data)
    % Compute mean, std, min, max of parameters for each distribution
    % Reads directly from Param_<DIST>_<param> columns
    
    var_names = data.Properties.VariableNames;
    param_cols = var_names(startsWith(var_names, 'Param_'));
    
    % Parse distribution names from column names
    dist_names = {};
    for i = 1:length(param_cols)
        parts = strsplit(param_cols{i}, '_');
        dist_name = parts{2};  % Param_GAMMA_shape -> GAMMA
        if ~ismember(dist_name, dist_names)
            dist_names{end+1} = dist_name;
        end
    end
    
    param_stats = struct('distribution', {}, 'mean', {}, 'std', {}, 'min', {}, 'max', {}, 'n_samples', {});
    
    for d = 1:length(dist_names)
        dist_name = dist_names{d};
        
        % Find all param columns for this distribution
        dist_cols = param_cols(startsWith(param_cols, ['Param_' dist_name '_']));
        
        if isempty(dist_cols)
            continue;
        end
        
        % Get rows for this distribution (non-NaN values)
        first_col = dist_cols{1};
        valid_mask = ~isnan(data.(first_col));
        n_samples = sum(valid_mask);
        
        if n_samples == 0
            continue;
        end
        
        stats_entry = struct();
        stats_entry.distribution = dist_name;
        stats_entry.n_samples = n_samples;
        
        mean_struct = struct();
        std_struct = struct();
        min_struct = struct();
        max_struct = struct();
        
        for c = 1:length(dist_cols)
            col_name = dist_cols{c};
            parts = strsplit(col_name, '_');
            param_name = parts{3};  % Param_GAMMA_shape -> shape
            
            values = data.(col_name)(valid_mask);
            
            mean_struct.(param_name) = mean(values);
            std_struct.(param_name) = std(values);
            min_struct.(param_name) = min(values);
            max_struct.(param_name) = max(values);
        end
        
        stats_entry.mean = mean_struct;
        stats_entry.std = std_struct;
        stats_entry.min = min_struct;
        stats_entry.max = max_struct;
        
        param_stats(end+1) = stats_entry;
    end
end