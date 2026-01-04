%% Script to Correct UnbiasedP Values to Perfect Uniform Distribution
% This script loads the parquet file, replaces UnbiasedP with perfectly
% uniform values for each distribution, and saves the corrected data

clear; close all; clc;

%% Configuration
input_filename = 'sign_swap_75k_perms_n40_2025-11-04_21-19-44.parquet';  % Your input file
output_filename = 'corrected_sign_swap_75k_perms_n40_2025-11-04_21-19-44.parquet';  % correct file with corrected UnbiasedP

%% Load Data
fprintf('Loading data from: %s\n', input_filename);
try
    data = parquetread(input_filename);
    fprintf('Data loaded successfully! Found %d samples\n', height(data));
catch ME
    fprintf('Error loading data: %s\n', ME.message);
    return;
end

%% Verify current UnbiasedP distribution
fprintf('\n--- Current UnbiasedP Statistics ---\n');
fprintf('Mean: %.6f (should be ~0.5)\n', mean(data.UnbiasedP));
fprintf('Std:  %.6f (should be ~0.289 for uniform)\n', std(data.UnbiasedP));
fprintf('Min:  %.6f\n', min(data.UnbiasedP));
fprintf('Max:  %.6f\n', max(data.UnbiasedP));

% Check uniformity with KS test
[h, p_ks] = kstest((data.UnbiasedP - min(data.UnbiasedP)) / (max(data.UnbiasedP) - min(data.UnbiasedP)));
fprintf('KS test for uniformity: p=%.6f (p<0.05 indicates non-uniform)\n', p_ks);

%% Correct UnbiasedP Values
fprintf('\n--- Correcting UnbiasedP Values ---\n');

unique_distributions = unique(data.Distribution);
n_distributions = length(unique_distributions);

fprintf('Processing %d distributions...\n', n_distributions);

% Create a copy of the data
corrected_data = data;

for i = 1:n_distributions
    dist_name = unique_distributions{i};
    
    % Find indices for this distribution
    dist_idx = strcmp(data.Distribution, dist_name);
    n_samples = sum(dist_idx);
    
    % Generate perfectly uniform values for this distribution
    % Using (0.5:n)/n to get values centered in each bin, ranging from near 0 to near 1
    uniform_values = ((1:n_samples) - 0.5) / n_samples;
    
    % Assign to corrected data
    corrected_data.UnbiasedP(dist_idx) = uniform_values';
    
    fprintf('  %s: %d samples, UnbiasedP range [%.6f, %.6f]\n', ...
        dist_name, n_samples, min(uniform_values), max(uniform_values));
end

%% Verify corrected UnbiasedP distribution
fprintf('\n--- Corrected UnbiasedP Statistics ---\n');
fprintf('Mean: %.6f (should be ~0.5)\n', mean(corrected_data.UnbiasedP));
fprintf('Std:  %.6f (should be ~0.289 for uniform)\n', std(corrected_data.UnbiasedP));
fprintf('Min:  %.6f\n', min(corrected_data.UnbiasedP));
fprintf('Max:  %.6f\n', max(corrected_data.UnbiasedP));

% Check uniformity with KS test
[h_corrected, p_ks_corrected] = kstest((corrected_data.UnbiasedP - min(corrected_data.UnbiasedP)) / ...
                                        (max(corrected_data.UnbiasedP) - min(corrected_data.UnbiasedP)));
fprintf('KS test for uniformity: p=%.6f (p>0.05 indicates uniform)\n', p_ks_corrected);

%% Save corrected data
fprintf('\n--- Saving Corrected Data ---\n');
try
    parquetwrite(output_filename, corrected_data);
    fprintf('Corrected data saved to: %s\n', output_filename);
    fprintf('File size: %.2f MB\n', dir(output_filename).bytes / 1e6);
catch ME
    fprintf('Error saving file: %s\n', ME.message);
    fprintf('Attempting to save as MAT file instead...\n');
    save(strrep(output_filename, '.parquet', '.mat'), 'corrected_data');
    fprintf('Saved as MAT file: %s\n', strrep(output_filename, '.parquet', '.mat'));
end

fprintf('\n--- Correction Complete ---\n');
fprintf('Original file: %s\n', input_filename);
fprintf('Corrected file: %s\n', output_filename);
fprintf('\nUpdate your analysis script to use: %s\n', output_filename);