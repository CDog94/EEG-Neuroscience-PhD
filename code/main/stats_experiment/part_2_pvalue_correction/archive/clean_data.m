%% Script to Set UnbiasedP = BiasedP for Symmetric Distributions
% Loads data, modifies UnbiasedP for symmetric distributions, saves result

clear; close all; clc;

%% Configuration
input_filename = 'two_sample.parquet';
output_filename = 'clean_two_sample.parquet';

%% Define symmetric distributions
symmetric_distributions = {'GAUSSIAN', 'UNIFORM', 'LAPLACE'};

%% Load data
fprintf('Loading data...\n');
data = parquetread(input_filename);

%% Modify UnbiasedP for symmetric distributions
for i = 1:length(symmetric_distributions)
    dist_pattern = symmetric_distributions{i};
    
    % Find rows matching this symmetric distribution
    is_current_dist = contains(data.Distribution, dist_pattern);
    
    % Set UnbiasedP = BiasedP for symmetric distributions
    data.UnbiasedP(is_current_dist) = data.BiasedP(is_current_dist);
    
    fprintf('Modified %d samples for %s\n', sum(is_current_dist), dist_pattern);
end

%% Save modified dataset
parquetwrite(output_filename, data);
fprintf('Saved to: %s\n', output_filename);

fprintf('Done.\n');