    % Optimized Polynomial Correction Model for Permutation Test P-values
% CPU-optimized version with fast data generation
% Supports PARQUET format for faster I/O
% FLEXIBLE: Choose between one-sample or paired two-sample tests
% MODIFIED: Flexible filtering - exclude selected distributions AND high skewness samples
% NEW: Option to use theoretical moments instead of sample-level moments

clear; close all; clc;

%% Parameters
nPValues = 50000000;
nParticipants = 150;         % Sample size per group
nPermutations = 3000;       % Number of permutations per test

% CHOOSE TEST TYPE: 'one-sample' or 'paired'
TEST_TYPE = 'one-sample';  % <-- CHANGE THIS TO SWITCH TEST TYPE

% FLEXIBLE FILTERING CONFIGURATION
DISTRIBUTIONS_TO_EXCLUDE = {};
DISTRIBUTIONS_TO_EXCLUDE = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T', 'EXPONENTIAL', 'LOGNORMAL'};  % <-- CHANGE THIS: List distributions to exclude, or use {} for none
DISTRIBUTIONS_TO_EXCLUDE = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T', 'EXPONENTIAL', 'LOGNORMAL'};

SKEWNESS_THRESHOLD = Inf;  % <-- CHANGE THIS: Exclude samples where |skewness| >= this value (set to Inf to disable)

USE_CALIBRATION = true;

% NEW: THEORETICAL MOMENTS OPTION
USE_THEORETICAL_MOMENTS = false;  % <-- CHANGE THIS: Set to true to use theoretical moments instead of sample moments

% OVERSAMPLING CONFIGURATION
USE_OVERSAMPLING = false;        % <-- CHANGE THIS: Set to true to enable oversampling
OVERSAMPLE_RANGE_MIN = 0.04;     % <-- CHANGE THIS: Minimum p-value for oversampling
OVERSAMPLE_RANGE_MAX = 0.06;     % <-- CHANGE THIS: Maximum p-value for oversampling  
OVERSAMPLE_FACTOR = 3.0; 

%% Set random seed for reproducibility
rng(42);

%% Load existing data or generate new
load_existing = true;  % Set to true to load existing data

% Determine filename based on test type
switch TEST_TYPE
    case 'one-sample'
        data_filename = 'sign_swap_corrected.parquet';
    case 'paired'
        data_filename = 'two_sample.parquet';
    otherwise
        error('Invalid TEST_TYPE. Choose: one-sample or paired');
end

if load_existing
    fprintf('Loading existing data from: %s\n', data_filename);
    data = loadParquetData(data_filename);
    fprintf('Data loaded successfully! Found %d samples\n', height(data));
        
    % Apply flexible filtering
    fprintf('\n=== APPLYING FLEXIBLE FILTERING ===\n');
    
    before_filtering = height(data);
    keep_idx = true(height(data), 1);
    
    % STEP 1: Distribution exclusion
    if ~isempty(DISTRIBUTIONS_TO_EXCLUDE)
        fprintf('Excluding selected distributions...\n');
        for i = 1:length(DISTRIBUTIONS_TO_EXCLUDE)
            dist_name = DISTRIBUTIONS_TO_EXCLUDE{i};
            dist_idx = strcmp(data.Distribution, dist_name);
            n_excluded = sum(dist_idx);
            if n_excluded > 0
                fprintf('  Excluding %s: %d samples\n', dist_name, n_excluded);
                keep_idx = keep_idx & ~dist_idx;
            else
                fprintf('  %s: no samples found to exclude\n', dist_name);
            end
        end
    else
        fprintf('No distributions selected for exclusion.\n');
    end
    
    after_dist_exclusion = sum(keep_idx);
    fprintf('After distribution exclusion: %d samples (%.1f%% remaining)\n', ...
            after_dist_exclusion, 100*after_dist_exclusion/before_filtering);
    
    % STEP 2: Skewness threshold filtering
    if isfinite(SKEWNESS_THRESHOLD)
        % Apply skewness threshold (exclude high absolute skewness)
        skewness_filter = abs(data.Skewness) >= SKEWNESS_THRESHOLD;
        keep_idx = keep_idx & skewness_filter;
    else
        fprintf('Skewness threshold disabled (set to Inf).\n');
    end
    
    % Apply the combined filter
    data = data(keep_idx, :);
    
    fprintf('\nFINAL FILTERING RESULTS:\n');
    fprintf('Original samples: %d\n', before_filtering);
    fprintf('Final samples: %d (%.1f%% remaining)\n', height(data), 100*height(data)/before_filtering);
    
    % Create filter description for titles
    filter_parts = {};
    if ~isempty(DISTRIBUTIONS_TO_EXCLUDE)
        filter_parts{end+1} = sprintf('Excluded: %s', strjoin(DISTRIBUTIONS_TO_EXCLUDE, ', '));
    end
    if isfinite(SKEWNESS_THRESHOLD)
        filter_parts{end+1} = sprintf('|Skewness| > %.1f', SKEWNESS_THRESHOLD);
    end
    
    if isempty(filter_parts)
        filter_description = 'No Filtering Applied';
    else
        filter_description = strjoin(filter_parts, ' & ');
    end
    
    % Show remaining distributions and their characteristics
    remaining_distributions = unique(data.Distribution);
    fprintf('\nRemaining distributions after filtering:\n');
    for i = 1:length(remaining_distributions)
        dist_name = remaining_distributions{i};
        n_samples = sum(strcmp(data.Distribution, dist_name));
        dist_data = data(strcmp(data.Distribution, dist_name), :);
        mean_skew = mean(dist_data.Skewness);
        std_skew = std(dist_data.Skewness);
        fprintf('  %s: %d samples (%.1f%%) - Skewness: μ=%.3f, σ=%.3f\n', ...
                dist_name, n_samples, 100*n_samples/height(data), mean_skew, std_skew);
    end
    
else

   
    %% Generate Training Data
    fprintf('Generating training data using FAST %s permutation tests...\n', upper(TEST_TYPE));
    fprintf('Target: %d p-0.alues, %d participants per group, %d permutations\n', ...
            nPValues, nParticipants, nPermutations);
    
    tic;
    data = generatePermutationDataFast(nPValues, nParticipants, nPermutations, TEST_TYPE);
    elapsed = toc;
    
    fprintf('Generated %d samples in %.2f seconds (%.1f samples/sec)\n', ...
            height(data), elapsed, height(data)/elapsed);
    
    % Apply flexible filtering (no critical region filter during data generation)
    fprintf('\nApplying flexible filters...\n');
    original_size = height(data);
    
    % Apply flexible filtering
    keep_idx = true(height(data), 1);
    
    % Distribution exclusion
    if ~isempty(DISTRIBUTIONS_TO_EXCLUDE)
        for i = 1:length(DISTRIBUTIONS_TO_EXCLUDE)
            dist_name = DISTRIBUTIONS_TO_EXCLUDE{i};
            dist_idx = strcmp(data.Distribution, dist_name);
            n_excluded = sum(dist_idx);
            if n_excluded > 0
                fprintf('  Excluding %s: %d samples\n', dist_name, n_excluded);
                keep_idx = keep_idx & ~dist_idx;
            end
        end
    end
    
    % Skewness threshold
    if isfinite(SKEWNESS_THRESHOLD)
        skewness_filter = abs(data.Skewness) >= SKEWNESS_THRESHOLD
        keep_idx = keep_idx & skewness_filter;
    end
    
    data = data(keep_idx, :);
    
    % Create filter description
    filter_parts = {};
    if ~isempty(DISTRIBUTIONS_TO_EXCLUDE)
        filter_parts{end+1} = sprintf('Excluded: %s', strjoin(DISTRIBUTIONS_TO_EXCLUDE, ', '));
    end
    if isfinite(SKEWNESS_THRESHOLD)
        filter_parts{end+1} = sprintf('|Skewness| >= %.1f', SKEWNESS_THRESHOLD);
    end
    
    if isempty(filter_parts)
        filter_description = 'No Filtering Applied';
    else
        filter_description = strjoin(filter_parts, ' & ');
    end
    
    fprintf('After filtering: %d samples (%.1f%% of original)\n', ...
            height(data), 100*height(data)/original_size);nDistributions
end

%% NEW: Apply theoretical moments if requested
if USE_THEORETICAL_MOMENTS
    fprintf('\n=== REPLACING SAMPLE MOMENTS WITH THEORETICAL MOMENTS ===\n');
    data = replaceWithTheoreticalMoments(data);
    filter_description = [filter_description ' + Theoretical Moments'];
end

%% Apply oversampling if requested
if USE_OVERSAMPLING
    fprintf('\n=== APPLYING OVERSAMPLING ===\n');
    data = oversamplePValueRange(data, OVERSAMPLE_RANGE_MIN, OVERSAMPLE_RANGE_MAX, OVERSAMPLE_FACTOR, true);
    filter_description = [filter_description sprintf(' + %.1fx Oversampled [%.3f,%.3f]', ...
                          OVERSAMPLE_FACTOR, OVERSAMPLE_RANGE_MIN, OVERSAMPLE_RANGE_MAX)];
end

%% Investigate moments vs biased p-values relationship
fprintf('\n=== INVESTIGATING MOMENTS VS BIASED P-VALUES (%s TEST, %s) ===\n', ...
        upper(TEST_TYPE), filter_description);

% Display summary statistics for remaining data
fprintf('\nSummary statistics for filtered data:\n');
fprintf('%-15s %8s %10s %10s %10s %10s\n', 'Distribution', 'Count', 'Mean Skew', 'Std Skew', 'Mean Kurt', 'Std Kurt');
fprintf('%s\n', repmat('-', 1, 73));

unique_dists = unique(data.Distribution);
for i = 1:length(unique_dists)
    dist_name = unique_dists{i};
    dist_idx = strcmp(data.Distribution, dist_name);
    dist_data = data(dist_idx, :);
    
    fprintf('%-15s %8d %10.4f %10.4f %10.4f %10.4f\n', ...
            dist_name, sum(dist_idx), ...
            mean(dist_data.Skewness), std(dist_data.Skewness), ...
            mean(dist_data.Kurtosis), std(dist_data.Kurtosis));
end

% Overall statistics
fprintf('%-15s %8d %10.4f %10.4f %10.4f %10.4f\n', ...
        'OVERALL', height(data), ...
        mean(data.Skewness), std(data.Skewness), ...
        mean(data.Kurtosis), std(data.Kurtosis));

%% Perform Train/Test Split and Model Training
fprintf('\nPerforming stratified 80/20 train/test split (%s)...\n', filter_description);
fprintf('Training full features model...\n');

[results, model] = trainTestSplit(data, USE_CALIBRATION);

%% Visualize Model
% Create the 3D surface plot
plotModelSurface(model, data, false, filter_description);  % Full range

%% Visualize Results
fprintf('\nVisualizing results...\n');

moments_type = ternary(USE_THEORETICAL_MOMENTS, 'Theoretical', 'Sample');
if USE_CALIBRATION && results.use_calibration && ~isempty(results.calibration_params)
   model_title = 'Polynomial Regressor: ';
    % Use calibrated predictions for visualization
    results_for_viz = results;
    results_for_viz.test_pred = results.test_pred_cal;
    results_for_viz.test_critical_idx = results.test_critical_idx_cal;
    results_for_viz.test_rmse = results.test_rmse_cal;
    results_for_viz.test_r2 = results.test_r2_cal;
    visualizeResults(data, results_for_viz, model, model_title);
else
    model_title = 'Polynomial Regressor: ';
    visualizeResults(data, results, model, model_title);
end

%% Final cleanup
close all;
clearvars;

%% ============================================================================
%% THEORETICAL MOMENTS FUNCTIONS
%% ============================================================================

function data_modified = replaceWithTheoreticalMoments(data)
    % Replace sample-level moments with theoretical moments from distributions
    % This creates a sanity check version using "perfect" theoretical moments
    % 
    % Input:
    %   data - table with columns: BiasedP, UnbiasedP, Mean, Variance, Skewness, Kurtosis, Distribution
    % 
    % Output:
    %   data_modified - same table but with theoretical moments replacing sample moments
    
    % Create a copy of the data
    data_modified = data;
    
    % Get unique distributions
    unique_distributions = unique(data.Distribution);
    
    fprintf('Processing %d distributions...\n', length(unique_distributions));
    
    % Process each distribution
    for i = 1:length(unique_distributions)
        dist_name = unique_distributions{i};
        dist_idx = strcmp(data.Distribution, dist_name);
        n_samples = sum(dist_idx);
        
        if n_samples == 0
            continue;
        end
        
        % Get theoretical moments for this distribution
        theoretical_moments = getTheoreticalMoments(dist_name);
        
        % Replace moments for all samples of this distribution
        data_modified.Mean(dist_idx) = theoretical_moments(1);
        data_modified.Variance(dist_idx) = theoretical_moments(2);
        data_modified.Skewness(dist_idx) = theoretical_moments(3);
        data_modified.Kurtosis(dist_idx) = theoretical_moments(4);
        
        fprintf('  %-12s: %d samples | Theoretical moments = [μ=%.4f, σ²=%.4f, γ₁=%.4f, γ₂=%.4f]\n', ...
                dist_name, n_samples, theoretical_moments(1), theoretical_moments(2), ...
                theoretical_moments(3), theoretical_moments(4));
    end
    
    fprintf('Replacement complete! All sample moments replaced with theoretical values.\n');
end

function moments = getTheoreticalMoments(distribution_name)
    % Calculate theoretical moments for each distribution
    % Returns [mean, variance, skewness, excess_kurtosis]
    % 
    % NOTE: Accounts for mean-centering done in generateDistributionSampleFast
    % For GAMMA, EXPONENTIAL, LOGNORMAL: original distribution is shifted by subtracting its mean
    
    switch upper(distribution_name)
        case 'NORMAL'
            % Normal(0, 1) - already centered at 0
            mean_val = 0;
            variance = 1;
            skewness = 0;
            excess_kurtosis = 0;  % Normal distribution has excess kurtosis = 0
            
        case 'UNIFORM'
            % Uniform(-0.5, 0.5) - already centered at 0
            mean_val = 0;  % (a + b)/2 = 0
            variance = (0.5 - (-0.5))^2 / 12;  % = 1/12 ≈ 0.0833
            skewness = 0;  % Symmetric distribution
            excess_kurtosis = -1.2;  % Uniform has excess kurtosis = -1.2
            
        case 'LAPLACE'
            % Laplace(0, 1) - already centered at 0
            mean_val = 0;
            variance = 2;  % 2 * scale^2 = 2 * 1^2 = 2
            skewness = 0;  % Symmetric distribution
            excess_kurtosis = 3;  % Laplace has excess kurtosis = 3
            
        case 'STUDENT_T'
            % Student's t with df=3 - already centered at 0
            df = 3;
            mean_val = 0;  % For df > 1
            if df > 2
                variance = df / (df - 2);  % = 3/(3-2) = 3
            else
                variance = Inf;
            end
            skewness = 0;  % Symmetric distribution for df > 3
            if df > 4
                excess_kurtosis = 6 / (df - 4);  % For df=3, this is undefined, but we use approximation
            else
                excess_kurtosis = 6;  % Large positive value for heavy tails when df <= 4
            end
            
        case 'GAMMA'
            % Gamma(shape=0.5, scale=1) then subtract mean to center at 0
            % Original: X ~ Gamma(0.5, 1), then Y = X - E[X] where E[X] = shape * scale = 0.5
            shape = 0.5; scale = 1;
            
            % After mean-centering: Y = X - 0.5
            mean_val = 0;  % By construction
            variance = shape * scale^2;  % Variance unchanged by translation: 0.5 * 1^2 = 0.5
            skewness = 2 / sqrt(shape);  % = 2 / sqrt(0.5) ≈ 2.828
            excess_kurtosis = 6 / shape;  % = 6 / 0.5 = 12
            
        case 'EXPONENTIAL'  
            % Exponential(rate=1) then subtract mean to center at 0
            % Original: X ~ Exp(1), then Y = X - E[X] where E[X] = 1/rate = 1
            rate = 1;
            
            % After mean-centering: Y = X - 1
            mean_val = 0;  % By construction
            variance = 1 / rate^2;  % Variance unchanged by translation: 1/1^2 = 1
            skewness = 2;  % Exponential always has skewness = 2
            excess_kurtosis = 6;  % Exponential always has excess kurtosis = 6
            
        case 'LOGNORMAL'
            % LogNormal(mu=0, sigma=0.5) then subtract mean to center at 0
            % Original: X ~ LogN(0, 0.5), then Y = X - E[X] where E[X] = exp(mu + sigma^2/2)
            mu = 0; sigma = 0.5;
            
            % After mean-centering: Y = X - original_mean
            mean_val = 0;  % By construction
            
            % For lognormal, Var(X) = (exp(σ²) - 1) * exp(2μ + σ²)
            variance = (exp(sigma^2) - 1) * exp(2*mu + sigma^2);  % ≈ 0.144
            
            % Skewness of lognormal: (exp(σ²) + 2) * sqrt(exp(σ²) - 1)
            skewness = (exp(sigma^2) + 2) * sqrt(exp(sigma^2) - 1);  % ≈ 1.75
            
            % Excess kurtosis of lognormal: exp(4σ²) + 2*exp(3σ²) + 3*exp(2σ²) - 6
            excess_kurtosis = exp(4*sigma^2) + 2*exp(3*sigma^2) + 3*exp(2*sigma^2) - 6;  % ≈ 8.87
            
        otherwise
            error('Unknown distribution: %s', distribution_name);
    end
    
    moments = [mean_val, variance, skewness, excess_kurtosis];
end

function result = ternary(condition, true_val, false_val)
    % Simple ternary operator helper function
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

%% ============================================================================
%% PERMUTATION TEST FUNCTIONS
%% ============================================================================

function p = permutation_test_one_sample_vectorized(group1, permIndices)
    % One-sample permutation test against zero using sign-flipping
    % group1: column vector of data
    % permIndices: matrix of size nPerms x n, containing 0s and 1s for sign flips
    % (pre-generate as: permIndices = rand(nPerms, n) > 0.5)
    %
    % Returns p-value based on permutation distribution of t-statistics
    
    n = length(group1);
    nPerms = size(permIndices, 1);
    
    % Observed one-sample t-statistic against zero
    mean_obs = mean(group1);
    std_obs = std(group1, 0); % Sample standard deviation
    
    if std_obs == 0
        % If no variance, can't compute t-statistic meaningfully
        p = 1;
        return;
    end
    obs_t = mean_obs / (std_obs / sqrt(n));
    
    % Convert 0/1 to -1/+1 for sign flipping
    signs = 2 * permIndices - 1; % 0 becomes -1, 1 stays 1
    
    % Apply sign flips to all permutations at once
    % Each row is a permutation
    flipped_data = signs .* repmat(group1(:)', nPerms, 1);
    
    % Compute means and standard deviations for all permutations
    perm_means = mean(flipped_data, 2);
    perm_stds = std(flipped_data, 0, 2); % Sample std along rows
    
    % Compute t-statistics for all permutations
    % Handle cases where std might be zero (though unlikely with sign flipping)
    valid_idx = perm_stds > 0;
    t_perm = zeros(nPerms, 1);
    t_perm(valid_idx) = perm_means(valid_idx) ./ (perm_stds(valid_idx) / sqrt(n));
    
    % Calculate two-sided p-value with +1 correction
    p = (sum(abs(t_perm) >= abs(obs_t)) + 1) / (nPerms + 1);
end

function p = permutation_test_paired_vectorized(group1, group2, swapIndices)
    % Vectorized PAIRED permutation test using paired t-statistic with within-pair swapping
    % group1, group2: column vectors of PAIRED data (must be same length)
    % swapIndices: binary matrix of size nPerms x n, where each row indicates
    %              which pairs to swap (1 = swap, 0 = keep original)
    
    n = length(group1);
    if length(group2) ~= n
        error('For paired permutation test, group1 and group2 must have the same length');
    end
    nPerms = size(swapIndices, 1);
    
    % Compute paired differences for observed data
    differences = group1(:) - group2(:);
    
    % Observed paired t-statistic
    obs_mean_diff = mean(differences);
    obs_se_diff = std(differences) / sqrt(n);
    if obs_se_diff > 0
        obs_t = obs_mean_diff / obs_se_diff;
    else
        obs_t = 0;  % Handle case where all differences are identical
    end
    
    % Create matrices for both groups repeated for each permutation
    group1_mat = repmat(group1(:)', nPerms, 1);  % nPerms x n
    group2_mat = repmat(group2(:)', nPerms, 1);  % nPerms x n
    
    % Convert swapIndices to logical for efficient indexing
    swap_mask = logical(swapIndices);
    
    % Initialize permuted groups (start with original assignments)
    perm_group1 = group1_mat;
    perm_group2 = group2_mat;
    
    % Perform swaps: where swapIndices is 1, swap the pair
    perm_group1(swap_mask) = group2_mat(swap_mask);
    perm_group2(swap_mask) = group1_mat(swap_mask);
    
    % Compute differences for all permutations
    perm_differences = perm_group1 - perm_group2;
    
    % Compute means and standard errors for all permutations
    perm_means = mean(perm_differences, 2);
    perm_std = std(perm_differences, 0, 2);
    perm_se = perm_std / sqrt(n);
    
    % Compute t-statistics for all permutations
    valid_se = perm_se > 0;
    t_perm = zeros(nPerms, 1);
    t_perm(valid_se) = perm_means(valid_se) ./ perm_se(valid_se);
    
    % Calculate two-sided p-value with +1 correction
    p = (sum(abs(t_perm) >= abs(obs_t)) + 1) / (nPerms + 1);
end

%% ============================================================================
%% FAST DATA GENERATION (FLEXIBLE TEST TYPE)
%% ============================================================================

function data = generatePermutationDataFast(nPValues, nParticipants, nPermutations, test_type)
    % OPTIMIZED: Generate p-values using specified test type
    
    % Start parallel pool if not already running
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('Starting parallel pool...\n');
        pool = parpool('local');
    else
        fprintf('Using existing parallel pool with %d workers\n', pool.NumWorkers);
    end
    
    % Define distributions
    distributions = [
        struct('name', 'EXPONENTIAL', 'params', [])
        struct('name', 'GAMMA',       'params', [])
        struct('name', 'NORMAL',      'params', [])
        struct('name', 'UNIFORM',     'params', [])
        struct('name', 'LAPLACE',     'params', [])
        struct('name', 'STUDENT_T',   'params', [])
        struct('name', 'LOGNORMAL',   'params', [])
    ];
     
    nDistributions = length(distributions);
    nPValuesPerDist = ceil(nPValues / nDistributions);
    actualTotal = nPValuesPerDist * nDistributions;
    
    fprintf('  Generating %d p-values per distribution (%d distributions)\n', ...
            nPValuesPerDist, nDistributions);
    fprintf('  Actual total: %d p-values\n', actualTotal);
    
    % Pre-allocate all arrays
    allBiasedP = zeros(actualTotal, 1);
    allMoments = zeros(actualTotal, 4);
    allDistLabels = zeros(actualTotal, 1);
    
    % Pre-generate permutation indices based on test type
    fprintf('  Pre-generating permutation indices for %s test...\n', test_type);
    tic;
    
    switch test_type
        case 'one-sample'
            % Sign-flipping indices (0/1 for sign flips)
            permIndices = rand(nPermutations, nParticipants) > 0.5;
            
        case 'paired'
            % Swap indices for paired test (0/1 for swaps)
            permIndices = rand(nPermutations, nParticipants) > 0.5;
            
        otherwise
            error('Invalid test_type: %s', test_type);
    end
    
    fprintf('    Done in %.2f seconds\n', toc);
    
    % Process each distribution
    idx = 1;
    for d = 1:nDistributions
        dist_info = distributions(d);
        fprintf('  Processing %s distribution (parallel, %s test): ', dist_info.name, upper(test_type));
        tic;
        
        % Pre-allocate for this distribution
        dist_pvals = zeros(nPValuesPerDist, 1);
        dist_moments = zeros(nPValuesPerDist, 4);
        
        % PARALLEL processing for each p-value
        parfor p = 1:nPValuesPerDist
            % Generate data and run appropriate test
            switch test_type
                case 'one-sample'
                    % One-sample test: single group against zero
                    group_data = generateDistributionSampleFast(nParticipants, dist_info);
                    pval = permutation_test_one_sample_vectorized(group_data, permIndices);
                    moments = calculateMomentsFast(group_data);
                    
                case 'paired'
                    % Paired test: experimental vs control (paired)
                    experimental_group = generateDistributionSampleFast(nParticipants, dist_info);
                    control_group = zeros(nParticipants, 1);  % Paired baseline
                    pval = permutation_test_paired_vectorized(experimental_group, control_group, permIndices);
                    differences = experimental_group - control_group;
                    moments = calculateMomentsFast(differences);
            end
            
            dist_pvals(p) = pval;
            dist_moments(p, :) = moments;
        end
        
        % Store results
        endIdx = idx + nPValuesPerDist - 1;
        allBiasedP(idx:endIdx) = dist_pvals;
        allMoments(idx:endIdx, :) = dist_moments;
        allDistLabels(idx:endIdx) = d;
        
        idx = endIdx + 1;
        
        elapsed = toc;
        fprintf(' Done! (%.2f seconds, %.1f p-values/sec)\n', elapsed, nPValuesPerDist/elapsed);
    end
    
    % Generate unbiased p-values
    fprintf('  Generating unbiased p-values...\n');
    allUnbiasedP = rand(actualTotal, 1);
    
    % Sort and pair by rank
    [biasedP_sorted, biasIdx] = sort(allBiasedP);
    unbiasedP_sorted = sort(allUnbiasedP);
    
    % Apply sorting to other arrays
    moments_sorted = allMoments(biasIdx, :);
    dist_labels_sorted = allDistLabels(biasIdx);
    
    % Convert numeric labels back to strings
    dist_names = {'EXPONENTIAL', 'GAMMA', 'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T', 'LOGNORMAL'};
    dist_labels_cell = dist_names(dist_labels_sorted)';
    
    % Create output table
    data = table(biasedP_sorted, unbiasedP_sorted, moments_sorted(:,1), moments_sorted(:,2), ...
                 moments_sorted(:,3), moments_sorted(:,4), dist_labels_cell, ...
        'VariableNames', {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', ...
                          'Skewness', 'Kurtosis', 'Distribution'});
    
    % Print summary
    fprintf('\n  Summary Statistics (%s TEST):\n', upper(test_type));
    fprintf('    Total samples: %d\n', height(data));
    fprintf('    Biased p-value range: [%.6f, %.6f]\n', min(biasedP_sorted), max(biasedP_sorted));
    fprintf('    Samples with p < 0.05: %d (%.1f%%)\n', ...
            sum(biasedP_sorted < 0.05), 100*mean(biasedP_sorted < 0.05));
    
    % Save data with test type in filename
    saveParquetData(data, sprintf('%s_permutation_pvalue_data', test_type), ...
                    nPValuesPerDist, nParticipants, nPermutations);
end

function moments = calculateMomentsFast(x)
    % Fast moment calculation with single pass
    n = length(x);
    
    % Mean and variance in one pass
    m = mean(x);
    x_centered = x - m;
    v = sum(x_centered.^2) / (n-1);
    
    % Higher moments only if needed
    if v > 0
        std_val = sqrt(v);
        x_standardized = x_centered / std_val;
        
        % Compute both at once
        x_std_3 = x_standardized.^3;
        x_std_4 = x_standardized.^4;
        
        s = mean(x_std_3) * (n/(n-1)) * (n/(n-2));  % Adjusted skewness
        k = mean(x_std_4) - 3;  % Excess kurtosis
    else
        s = 0;
        k = 0;
    end
    
    moments = [m, v, s, k];
end

function sample = generateDistributionSampleFast(n_total, dist_info)
    % Fast distribution sampling with optimization
    
    switch dist_info.name
        case 'NORMAL'
            mean_val = 0;
            std_val = 1;
            sample = normrnd(mean_val, std_val, n_total, 1) - mean_val;
            
        case 'LAPLACE'
            location = 0;
            scale = 1;
            u = rand(n_total, 1) - 0.5;
            sample = location - scale * sign(u) .* log(1 - 2 * abs(u));
            
        case 'STUDENT_T'
            df = 3;
            sample = trnd(df, n_total, 1);
            
        case 'UNIFORM'
            min_val = -0.5;
            max_val = 0.5;
            sample = min_val + (max_val - min_val) * rand(n_total, 1);
            
        case 'GAMMA'
            shape = 0.5;
            scale = 1;
            mean_gamma = shape * scale;
            sample = gamrnd(shape, scale, n_total, 1) - mean_gamma;
            
        case 'EXPONENTIAL'
            rate = 1;
            mean_exp = 1/rate;
            sample = exprnd(1/rate, n_total, 1) - mean_exp;
            
        case 'LOGNORMAL'
            mu = 0;
            sigma = 0.5;
            mean_lognorm = exp(mu + sigma^2/2);
            sample = lognrnd(mu, sigma, n_total, 1) - mean_lognorm;
            
        otherwise
            error('Unknown distribution: %s', dist_info.name);
    end
end

%% ============================================================================
%% PARQUET I/O FUNCTIONS
%% ============================================================================

function data = loadParquetData(filename)
    % Load data from Parquet file
    [~, ~, ext] = fileparts(filename);
    
    switch lower(ext)
        case '.parquet'
            fprintf('Loading Parquet file: %s\n', filename);
            full_data = parquetread(filename);
            
            core_features = {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Distribution'};
            
            available_features = {};
            for i = 1:length(core_features)
                if ismember(core_features{i}, full_data.Properties.VariableNames)
                    available_features{end+1} = core_features{i};
                end
            end
            
            data = full_data(:, available_features);
            fprintf('Loaded %d samples with features: %s\n', height(data), strjoin(available_features, ', '));
            
        case '.mat'
            fprintf('Loading MAT file: %s\n', filename);
            loaded = load(filename);
            data = loaded.data;
            
        otherwise
            error('Unsupported file format: %s. Use .parquet or .mat', ext);
    end
end

function saveParquetData(data, base_filename, nPValuesPerDist, nParticipants, nPermutations)
    % Save data in Parquet format
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    
    % Add metadata columns
    data.nPValuesPerDist = repmat(nPValuesPerDist, height(data), 1);
    data.nParticipants = repmat(nParticipants, height(data), 1);
    data.nPermutations = repmat(nPermutations, height(data), 1);
    data.timestamp = repmat({timestamp}, height(data), 1);
    
    % Save as Parquet
    parquet_filename = sprintf('%s_%s.parquet', base_filename, timestamp);
    fprintf('Saving data to Parquet: %s\n', parquet_filename);
    tic;
    parquetwrite(parquet_filename, data);
    parquet_time = toc;
    fprintf('Parquet save completed in %.2f seconds\n', parquet_time);
    
    % Show file size
    parquet_info = dir(parquet_filename);
    fprintf('File size: %.2f MB\n', parquet_info.bytes / (1024*1024));
end

%% ============================================================================
%% MODEL TRAINING AND EVALUATION
%% ============================================================================

function [results, model] = trainTestSplit(data, use_calibration)
    % Perform stratified train/calibration/test split and train model
    % If use_calibration is true: 60/20/20 split for train/calibration/test
    % If use_calibration is false: 80/20 split for train/test
    
    if nargin < 2
        use_calibration = false;
    end
    
    unique_distributions = unique(data.Distribution);
    nDistributions = length(unique_distributions);
    
    % Initialize indices
    train_idx = false(height(data), 1);
    cal_idx = false(height(data), 1);
    test_idx = false(height(data), 1);
    
    if use_calibration
        fprintf('Performing stratified 60/20/20 train/calibration/test split...\n');
        train_pct = 0.6;
        cal_pct = 0.2;
        test_pct = 0.2;
    else
        fprintf('Performing stratified 80/20 train/test split (no calibration)...\n');
        train_pct = 0.8;
        cal_pct = 0.0;
        test_pct = 0.2;
    end
    
    % Stratified split by distribution
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_indices = find(strcmp(data.Distribution, dist_name));
        n_samples = length(dist_indices);
        
        % Randomly permute indices
        perm_indices = dist_indices(randperm(n_samples));
        
        % Calculate split sizes
        n_train = round(train_pct * n_samples);
        if use_calibration
            n_cal = round(cal_pct * n_samples);
            n_test = n_samples - n_train - n_cal;
        else
            n_cal = 0;
            n_test = n_samples - n_train;
        end
        
        % Assign indices
        train_idx(perm_indices(1:n_train)) = true;
        if use_calibration && n_cal > 0
            cal_idx(perm_indices(n_train+1:n_train+n_cal)) = true;
            test_idx(perm_indices(n_train+n_cal+1:end)) = true;
            fprintf('Distribution %s: %d train, %d calibration, %d test\n', dist_name, n_train, n_cal, n_test);
        else
            test_idx(perm_indices(n_train+1:end)) = true;
            fprintf('Distribution %s: %d train, %d test\n', dist_name, n_train, n_test);
        end
    end
    
    % Split data
    train_data = data(train_idx, :);
    test_data = data(test_idx, :);
    if use_calibration
        cal_data = data(cal_idx, :);
        fprintf('\nTotal: %d train, %d calibration, %d test samples\n', height(train_data), height(cal_data), height(test_data));
    else
        cal_data = [];
        fprintf('\nTotal: %d train samples, %d test samples\n', height(train_data), height(test_data));
    end
    
    % Create polynomial features for training
    X_train = [train_data.BiasedP, train_data.Mean, train_data.Skewness, train_data.Variance, train_data.Kurtosis];
    X_train_poly = createPolynomialFeatures(X_train);
    y_train = train_data.UnbiasedP;
    
    % Train model
    fprintf('Training polynomial regression model...\n');
    model = fitlm(X_train_poly, y_train);
    
    % Predictions on training set
    y_train_pred = predict(model, X_train_poly);
    
    % Learn calibration if enabled
    calibration_params = [];
    if use_calibration && ~isempty(cal_data)
        fprintf('\n=== LEARNING CALIBRATION ===\n');
        X_cal = [cal_data.BiasedP, cal_data.Mean, cal_data.Skewness, cal_data.Variance, cal_data.Kurtosis];
        X_cal_poly = createPolynomialFeatures(X_cal);
        y_cal = cal_data.UnbiasedP;
        y_cal_pred = predict(model, X_cal_poly);
        
        calibration_params = learnCalibration(y_cal, y_cal_pred);
    end
    
    % Create test features and predictions
    X_test = [test_data.BiasedP, test_data.Mean, test_data.Skewness, test_data.Variance, test_data.Kurtosis];
    X_test_poly = createPolynomialFeatures(X_test);
    y_test = test_data.UnbiasedP;
    
    % Test set predictions (uncalibrated)
    y_test_pred_raw = predict(model, X_test_poly);
    
    % Apply calibration to test predictions if enabled
    if use_calibration && ~isempty(calibration_params)
        fprintf('\n=== APPLYING CALIBRATION TO TEST SET ===\n');
        y_test_pred_cal = applyCalibration(y_test_pred_raw, calibration_params);
    else
        y_test_pred_cal = y_test_pred_raw;  % No calibration
    end
    
    % Evaluate performance - Training set (always uncalibrated)
    train_critical_idx = (y_train <= 0.05) & (y_train_pred <= 0.05);
    if sum(train_critical_idx) > 0
        train_rmse = sqrt(mean((y_train_pred(train_critical_idx) - y_train(train_critical_idx)).^2));
        train_r2 = 1 - sum((y_train(train_critical_idx) - y_train_pred(train_critical_idx)).^2) / ...
                   sum((y_train(train_critical_idx) - mean(y_train(train_critical_idx))).^2);
        train_mae = mean(abs(y_train_pred(train_critical_idx) - y_train(train_critical_idx)));
    else
        train_rmse = NaN; train_r2 = NaN; train_mae = NaN;
    end
    
    % Evaluate performance - Test set (uncalibrated)
    test_critical_idx_raw = (y_test <= 0.05) & (y_test_pred_raw <= 0.05);
    if sum(test_critical_idx_raw) > 0
        test_rmse_raw = sqrt(mean((y_test_pred_raw(test_critical_idx_raw) - y_test(test_critical_idx_raw)).^2));
        test_r2_raw = 1 - sum((y_test(test_critical_idx_raw) - y_test_pred_raw(test_critical_idx_raw)).^2) / ...
                      sum((y_test(test_critical_idx_raw) - mean(y_test(test_critical_idx_raw))).^2);
        test_mae_raw = mean(abs(y_test_pred_raw(test_critical_idx_raw) - y_test(test_critical_idx_raw)));
    else
        test_rmse_raw = NaN; test_r2_raw = NaN; test_mae_raw = NaN;
    end
    
    % *** FIXED FPR CALCULATION - PROFESSOR'S METHOD ***
    % Under null hypothesis simulation, all samples are true negatives
    % FPR = proportion of all predictions that are ≤ 0.05
    test_fpr_raw = sum(y_test_pred_raw <= 0.05) / length(y_test_pred_raw);
    
    % Evaluate performance - Test set (calibrated, if applicable)
    if use_calibration && ~isempty(calibration_params)
        test_critical_idx_cal = (y_test <= 0.05) & (y_test_pred_cal <= 0.05);
        if sum(test_critical_idx_cal) > 0
            test_rmse_cal = sqrt(mean((y_test_pred_cal(test_critical_idx_cal) - y_test(test_critical_idx_cal)).^2));
            test_r2_cal = 1 - sum((y_test(test_critical_idx_cal) - y_test_pred_cal(test_critical_idx_cal)).^2) / ...
                          sum((y_test(test_critical_idx_cal) - mean(y_test(test_critical_idx_cal))).^2);
            test_mae_cal = mean(abs(y_test_pred_cal(test_critical_idx_cal) - y_test(test_critical_idx_cal)));
        else
            test_rmse_cal = NaN; test_r2_cal = NaN; test_mae_cal = NaN;
        end
        
        % *** FIXED FPR CALCULATION FOR CALIBRATED MODEL ***
        % Under null hypothesis simulation, all samples are true negatives
        % FPR = proportion of all predictions that are ≤ 0.05
        test_fpr_cal = sum(y_test_pred_cal <= 0.05) / length(y_test_pred_cal);
    else
        test_rmse_cal = test_rmse_raw;
        test_r2_cal = test_r2_raw;
        test_mae_cal = test_mae_raw;
        test_fpr_cal = test_fpr_raw;
        test_critical_idx_cal = test_critical_idx_raw;
    end
    
    % Store results
    results = struct();
    results.train_data = train_data;
    results.test_data = test_data;
    results.cal_data = cal_data;
    results.train_pred = y_train_pred;
    results.test_pred_raw = y_test_pred_raw;
    results.test_pred_cal = y_test_pred_cal;
    results.train_critical_idx = train_critical_idx;
    results.test_critical_idx_raw = test_critical_idx_raw;
    results.test_critical_idx_cal = test_critical_idx_cal;
    results.calibration_params = calibration_params;
    results.use_calibration = use_calibration;
    
    % Performance metrics
    results.train_rmse = train_rmse;
    results.train_r2 = train_r2;
    results.train_mae = train_mae;
    results.test_rmse_raw = test_rmse_raw;
    results.test_r2_raw = test_r2_raw;
    results.test_mae_raw = test_mae_raw;
    results.test_fpr_raw = test_fpr_raw;
    results.test_rmse_cal = test_rmse_cal;
    results.test_r2_cal = test_r2_cal;
    results.test_mae_cal = test_mae_cal;
    results.test_fpr_cal = test_fpr_cal;
    
    results.X_train_poly = X_train_poly;
    results.X_test_poly = X_test_poly;
    
    % Print results
    fprintf('\n========== MODEL PERFORMANCE (CRITICAL REGION ≤ 0.05) ==========\n');
    fprintf('Training Set (n=%d critical samples, uncalibrated):\n', sum(train_critical_idx));
    fprintf('  RMSE: %.6f, R²: %.4f, MAE: %.6f\n', train_rmse, train_r2, train_mae);
    
    fprintf('Test Set - UNCALIBRATED (n=%d critical samples):\n', sum(test_critical_idx_raw));
    fprintf('  RMSE: %.6f, R²: %.4f, MAE: %.6f, FPR: %.4f\n', test_rmse_raw, test_r2_raw, test_mae_raw, test_fpr_raw);
    
    if use_calibration && ~isempty(calibration_params)
        fprintf('Test Set - CALIBRATED (n=%d critical samples):\n', sum(test_critical_idx_cal));
        fprintf('  RMSE: %.6f, R²: %.4f, MAE: %.6f, FPR: %.4f\n', test_rmse_cal, test_r2_cal, test_mae_cal, test_fpr_cal);
        
        % Show improvement
        rmse_improvement = ((test_rmse_raw - test_rmse_cal) / test_rmse_raw) * 100;
        fpr_change = ((test_fpr_cal - test_fpr_raw) / test_fpr_raw) * 100;
        fprintf('CALIBRATION IMPACT:\n');
        fprintf('  RMSE change: %.2f%% (%.6f → %.6f)\n', rmse_improvement, test_rmse_raw, test_rmse_cal);
        fprintf('  FPR change: %.2f%% (%.4f → %.4f)\n', fpr_change, test_fpr_raw, test_fpr_cal);
    end
    
    fprintf('Model R² (full training data): %.4f\n', model.Rsquared.Ordinary);
    
    fprintf('\n========== MODEL COEFFICIENTS ==========\n');
    printModelCoefficients(model);
    fprintf('=======================================\n');
end

function printModelCoefficients(model)
    % Print model coefficients with feature names
    feature_names = {
        'Intercept',
        'BiasedP',
        'Mean', 
        'Skewness',
        'Variance',
        'Kurtosis',
        'BiasedP²',
        'BiasedP³',
        'Mean²',
        'BiasedP×Mean',
        'Skewness²',
        'BiasedP×Skewness',
        'Variance²',
        'BiasedP×Variance',
        'Kurtosis²',
        'BiasedP×Kurtosis'
    };
    
    coeffs = model.Coefficients;
    
    fprintf('    %-18s %12s %12s %12s %12s\n', 'Feature', 'Coefficient', 'SE', 'tStat', 'pValue');
    fprintf('    %s\n', repmat('-', 1, 70));
    
    for i = 1:height(coeffs)
        fprintf('    %-18s %12.6f %12.6f %12.3f %12.6f\n', ...
                feature_names{i}, coeffs.Estimate(i), coeffs.SE(i), ...
                coeffs.tStat(i), coeffs.pValue(i));
    end
end

function X_poly = createPolynomialFeatures(X)
    % Create polynomial features
    % X columns: [BiasedP, Mean, Skewness, Variance, Kurtosis]
    X_poly = [X, ...                              % Linear terms (5 features)
             X(:,1).^2, X(:,1).^3, ...           % BiasedP quadratic & cubic
             X(:,2).^2, ...                      % Mean squared
             X(:,1).*X(:,2), ...                 % BiasedP * Mean
             X(:,3).^2, ...                      % Skewness squared
             X(:,1).*X(:,3), ...                 % BiasedP * Skewness
             X(:,4).^2, ...                      % Variance squared
             X(:,1).*X(:,4), ...                 % BiasedP * Variance
             X(:,5).^2, ...                      % Kurtosis squared
             X(:,1).*X(:,5)];                    % BiasedP * Kurtosis
end

function visualizeResults(data, results, model, model_name)
    % Visualize train/test results with focus on critical p-value region
    
    if nargin < 4
        model_name = 'Model';
    end
    
    % Determine which prediction fields to use
    if results.use_calibration && isfield(results, 'test_pred_cal') && ~isempty(results.calibration_params)
        test_predictions = results.test_pred_cal;
        test_critical_idx = results.test_critical_idx_cal;
        test_rmse = results.test_rmse_cal;
        test_r2 = results.test_r2_cal;
        test_fpr = results.test_fpr_cal;
    else
        test_predictions = results.test_pred_raw;
        test_critical_idx = results.test_critical_idx_raw;
        test_rmse = results.test_rmse_raw;
        test_r2 = results.test_r2_raw;
        test_fpr = results.test_fpr_raw;
    end
    
    unique_distributions = unique(data.Distribution);
    nDistributions = length(unique_distributions);
    colors = lines(nDistributions);
    
    % Sample 5% of data for plotting speed
    n_test = height(results.test_data);
    sample_size = round(0.05 * n_test);  % 5% sample
    sample_idx = randperm(n_test, sample_size);
    
    fprintf('Using %d samples (5%%) for visualization, full %d samples for statistics\n', sample_size, n_test);
    
    % Create sampled data for plotting
    test_data_sample = results.test_data(sample_idx, :);
    test_predictions_sample = test_predictions(sample_idx);
    
    % Create figure
    figure('Position', [50, 50, 2400, 1400]);
    
    % Main QQ plot focused on critical region (0-0.05)
    subplot(2, 3, [1, 2]); 
    
    % Filter test data to critical region (using sampled data for plotting)
    critical_idx_sample = (test_data_sample.UnbiasedP <= 0.075) & (test_predictions_sample <= 0.075);
    y_true_critical = test_data_sample.UnbiasedP(critical_idx_sample);
    y_pred_critical = test_predictions_sample(critical_idx_sample);
    test_dist_critical = test_data_sample.Distribution(critical_idx_sample);
    
    % Plot by distribution with FLIPPED AXES and SMALLER POINTS
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(test_dist_critical, dist_name);
        
        if sum(dist_idx) > 0
            scatter(y_pred_critical(dist_idx), y_true_critical(dist_idx), 5, colors(i,:), 'filled', ...
                   'MarkerFaceAlpha', 0.4, 'DisplayName', dist_name);
            hold on;
        end
    end
    
    % Perfect prediction line
    plot([0, 0.075], [0, 0.075], 'r--', 'LineWidth', 3, 'DisplayName', 'Perfect Prediction');
    
    % ADD REFERENCE LINES AT 0.05 THRESHOLD
    % Vertical line at x = 0.05 (predicted p-value threshold)
    xline(0.05, 'r-', 'LineWidth', 2, 'DisplayName', 'α = 0.05 Threshold');
    
    % Horizontal line at y = 0.05 (desired p-value threshold) 
    yline(0.05, 'r-', 'LineWidth', 2, 'HandleVisibility', 'off'); % Don't show in legend twice


    xlabel('Predicted Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Desired Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('QQ Plot: Critical Region (0-0.05)', 'FontSize', 14, 'FontWeight', 'bold');
    xlim([0, 0.075]);
    ylim([0, 0.075]);
    grid on;
    axis square;
    legend('Location', 'southeast', 'FontSize', 10);
    
    % Full range QQ plot with FLIPPED AXES and SMALLER POINTS (using sampled data)
    subplot(2, 3, 3);
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(test_data_sample.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            scatter(test_predictions_sample(dist_idx), test_data_sample.UnbiasedP(dist_idx), 5, colors(i,:), 'filled', ...
                   'MarkerFaceAlpha', 0.4, 'DisplayName', dist_name);
            hold on;
        end
    end
    plot([0, 1], [0, 1], 'r--', 'LineWidth', 3, 'DisplayName', 'Perfect Prediction');
    
    xlabel('Predicted Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Desired Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('QQ Plot: Full Range', 'FontSize', 14, 'FontWeight', 'bold');
    xlim([0, 1]);
    ylim([0, 1]);
    grid on;
    axis square;
    legend('Location', 'southeast', 'FontSize', 10);
    
    % Residual plot (WITH FLIPPED AXES - using sampled data)
    subplot(2, 3, 4);
    residuals_sample = test_predictions_sample - test_data_sample.UnbiasedP;
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(test_data_sample.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            scatter(residuals_sample(dist_idx), test_data_sample.UnbiasedP(dist_idx), 40, colors(i,:), 'filled', ...
                   'MarkerFaceAlpha', 0.4, 'DisplayName', dist_name);
            hold on;
        end
    end
    xline(0, 'r--', 'LineWidth', 2);  % Vertical line at zero residuals
    xlabel('Residuals (Pred - True)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Desired Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('Residual Plot', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    
    % Distribution characteristics (using sampled data)
    subplot(2, 3, 5);
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(test_data_sample.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            test_data_dist = test_data_sample(dist_idx, :);
            valid_idx = ~isnan(test_data_dist.Skewness) & ~isnan(test_data_dist.Kurtosis) & ...
                       ~isinf(test_data_dist.Skewness) & ~isinf(test_data_dist.Kurtosis) & ...
                       abs(test_data_dist.Skewness) < 10 & abs(test_data_dist.Kurtosis) < 50;
            
            if sum(valid_idx) > 0
                scatter(test_data_dist.Skewness(valid_idx), test_data_dist.Kurtosis(valid_idx), ...
                       5, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.4, 'DisplayName', dist_name);
                hold on;
            end
        end
    end
    xlabel('Skewness', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Kurtosis', 'FontSize', 12, 'FontWeight', 'bold');
    title('Distribution Characteristics', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    
    % Performance metrics by distribution: RMSE (critical region) and FPR (full range)
    subplot(2, 3, 6);
    dist_rmse = zeros(nDistributions, 1);
    dist_n_critical = zeros(nDistributions, 1);
    dist_fpr = zeros(nDistributions, 1);
    dist_n_true_negatives = zeros(nDistributions, 1);
    
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(results.test_data.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            y_true_dist = results.test_data.UnbiasedP(dist_idx);
            y_pred_dist = test_predictions(dist_idx);
            
            % Calculate POPULATION FPR (across entire distribution sample)
            false_positives = sum((y_true_dist > 0.05) & (y_pred_dist <= 0.05));
            total_samples = length(y_true_dist);
            dist_n_total(i) = total_samples;
            
            % Under null hypothesis, all samples are true negatives
            % FPR = proportion of predictions ≤ 0.05
            total_samples = length(y_pred_dist);
            if total_samples > 0
                dist_fpr(i) = sum(y_pred_dist <= 0.05) / total_samples;
            else
                dist_fpr(i) = NaN;
            end
                                    
            % Calculate RMSE on CRITICAL REGION (0-0.05)
            critical_idx_dist = (y_true_dist <= 0.05) & (y_pred_dist <= 0.05);
            dist_n_critical(i) = sum(critical_idx_dist);
            
            if sum(critical_idx_dist) > 0
                y_true_critical_dist = y_true_dist(critical_idx_dist);
                y_pred_critical_dist = y_pred_dist(critical_idx_dist);
                dist_rmse(i) = sqrt(mean((y_pred_critical_dist - y_true_critical_dist).^2));
            else
                dist_rmse(i) = NaN;
            end
        end
    end
    
    % Create grouped bar chart for RMSE and FPR
    bar_data = [dist_rmse, dist_fpr];
    b = bar(bar_data, 'grouped');
    
    % Handle both single and multiple distribution cases
    if length(b) == 1
        % Single distribution case - bar returns one object with multiple series
        b.FaceColor = 'flat';
        b.CData = [0.3 0.6 0.9; 0.9 0.3 0.3];  % First row for RMSE, second for FPR
    else
        % Multiple distribution case - bar returns array of objects
        b(1).FaceColor = [0.3 0.6 0.9];  % RMSE bars
        b(2).FaceColor = [0.9 0.3 0.3];  % FPR bars
    end
    
    set(gca, 'XTickLabel', unique_distributions);
    xtickangle(45);
    ylabel('Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('Performance Metrics by Distribution', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'RMSE', 'FPR'}, 'Location', 'northwest');
    grid on;
    
    % Add annotations with sample sizes
    for i = 1:nDistributions
        if ~isnan(dist_rmse(i))
            text(i - 0.15, dist_rmse(i) + max(dist_rmse)*0.02, sprintf('n=%d', dist_n_critical(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end
        if ~isnan(dist_fpr(i))
            text(i + 0.15, dist_fpr(i) + max(dist_fpr)*0.02, sprintf('n=%d', dist_n_total(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    
    % Calculate overall FPR on FULL RANGE (0-1)
    overall_fpr = sum(test_predictions <= 0.05) / length(test_predictions);
    
    % Main title with FPR included
    sgtitle(sprintf('%s Performance: Test RMSE=%.6f, Test R²=%.4f, Overall FPR=%.4f', ...
            model_name, test_rmse, test_r2, overall_fpr), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % Save and close
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = sprintf('results_plot_%s.png', timestamp);
    saveas(gcf, filename);
    fprintf('Saved: %s\n', filename);
    close(gcf);
end

function plotModelSurface(model, data, focus_critical_region, filter_description)
    % Create 3D surface plot showing how the model transforms BiasedP to UnbiasedP
    
    if nargin < 3
        focus_critical_region = false;
    end
    if nargin < 4
        filter_description = '';
    end
    
    % Set up the grid for BiasedP and Skewness
    if focus_critical_region
        biased_p_range = linspace(0, 0.05, 50);
        plot_title_suffix = ' (Critical Region)';
    else
        biased_p_range = linspace(0, 1, 50);
        plot_title_suffix = ' (Full Range)';
    end
    
    % Determine reasonable range for skewness from the data
    skewness_min = prctile(data.Skewness, 5);
    skewness_max = prctile(data.Skewness, 95);
    skewness_range = linspace(skewness_min, skewness_max, 50);
    
    % Fix other variables at their median values
    median_mean = median(data.Mean);
    median_variance = median(data.Variance);
    median_kurtosis = median(data.Kurtosis);
    
    % Create meshgrid
    [BiasedP_grid, Skewness_grid] = meshgrid(biased_p_range, skewness_range);
    
    % Initialize prediction grid
    UnbiasedP_grid = zeros(size(BiasedP_grid));
    
    % Generate predictions for each grid point
    fprintf('Generating surface predictions...\n');
    for i = 1:numel(BiasedP_grid)
        X_point = [BiasedP_grid(i), median_mean, Skewness_grid(i), median_variance, median_kurtosis];
        X_poly = createPolynomialFeatures(X_point);
        UnbiasedP_grid(i) = predict(model, X_poly);
    end
    
    % Clip predictions to [0, 1] range
    UnbiasedP_grid = max(0, min(1, UnbiasedP_grid));
    
    % Create the figure
    figure('Position', [100, 100, 1400, 600]);
    
    % Subplot 1: 3D Surface Plot
    subplot(1, 2, 1);
    surf(BiasedP_grid, Skewness_grid, UnbiasedP_grid, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
    colormap('parula');
    colorbar;
    
    xlabel('Biased P-value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Skewness', 'FontSize', 12, 'FontWeight', 'bold');
    zlabel('Predicted Unbiased P-value', 'FontSize', 12, 'FontWeight', 'bold');
    title(['Model Surface: BiasedP × Skewness → UnbiasedP' plot_title_suffix], ...
          'FontSize', 14, 'FontWeight', 'bold');
    
    lighting gouraud;
    light('Position', [1, 1, 1]);
    light('Position', [-1, -1, 0.5]);
    
    view(45, 30);
    grid on;
    
    % Add reference plane at z = x (perfect calibration)
    hold on;
    [X_ref, Y_ref] = meshgrid(biased_p_range, [skewness_min, skewness_max]);
    Z_ref = repmat(biased_p_range, 2, 1);
    surf(X_ref, Y_ref, Z_ref, 'FaceColor', 'red', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    
    % Subplot 2: Contour Plot (Top-down view)
    subplot(1, 2, 2);
    
    % Calculate correction amount (UnbiasedP - BiasedP)
    Correction_grid = UnbiasedP_grid - BiasedP_grid;
    
    % Create filled contour plot
    contourf(BiasedP_grid, Skewness_grid, Correction_grid, 20);
    
    % Use a diverging colormap
    n = 256;
    blue_to_white = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1)];
    white_to_red = [ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)'];
    custom_colormap = [blue_to_white; white_to_red];
    colormap(gca, custom_colormap);
    
    c = colorbar;
    c.Label.String = 'Correction (Unbiased - Biased)';
    c.Label.FontSize = 11;
    c.Label.FontWeight = 'bold';
    
    % Add zero correction line
    hold on;
    contour(BiasedP_grid, Skewness_grid, Correction_grid, [0 0], 'k-', 'LineWidth', 2);
    
    xlabel('Biased P-value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Skewness', 'FontSize', 12, 'FontWeight', 'bold');
    title(['Correction Surface: (UnbiasedP - BiasedP)' plot_title_suffix], ...
          'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    % Overall title with filter description
    if ~isempty(filter_description)
        sgtitle(sprintf('Model Behavior: How Skewness Affects P-value Correction (%s)', filter_description), ...
                'FontSize', 16, 'FontWeight', 'bold');
    else
        sgtitle('Model Behavior: How Skewness Affects P-value Correction', ...
                'FontSize', 16, 'FontWeight', 'bold');
    end
    
    fprintf('\n========== INTERPRETATION GUIDE ==========\n');
    fprintf('3D Surface Plot:\n');
    fprintf('  - Surface height shows predicted unbiased p-value\n');
    fprintf('  - Red transparent plane shows perfect calibration (no correction)\n');
    fprintf('  - Above red plane: model increases p-value (conservative)\n');
    fprintf('  - Below red plane: model decreases p-value (liberal)\n\n');
    fprintf('Contour Plot:\n');
    fprintf('  - Blue regions: negative correction (UnbiasedP < BiasedP)\n');
    fprintf('  - Red regions: positive correction (UnbiasedP > BiasedP)\n');
    fprintf('  - Black line: zero correction boundary\n');
    fprintf('  - Contour lines show equal correction amounts\n');
    fprintf('==========================================\n');

    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = sprintf('surface_plot_%s.png', timestamp);
    saveas(gcf, filename);
    fprintf('Saved: %s\n', filename);
    close(gcf);
end

function calibration_params = learnCalibration(true_values, predicted_values, cal_range_min, cal_range_max)
    % Learn calibration parameters using linear rescaling within a specified range
    % INPUTS:
    %   true_values: Ground truth values
    %   predicted_values: Predicted values to calibrate
    %   cal_range_min: Minimum value of calibration range (default: 0)
    %   cal_range_max: Maximum value of calibration range (default: 0.10)
    
    % Set default range if not provided
    if nargin < 3 || isempty(cal_range_min)
        cal_range_min = 0.04;
    end
    if nargin < 4 || isempty(cal_range_max)
        cal_range_max = 0.10;
    end
    
    % Validate range
    if cal_range_min >= cal_range_max
        error('cal_range_min (%.3f) must be less than cal_range_max (%.3f)', cal_range_min, cal_range_max);
    end
    
    fprintf('Learning calibration in range [%.3f, %.3f]...\n', cal_range_min, cal_range_max);
    
    % Focus on specified range where bias needs correction
    range_idx = (true_values >= cal_range_min & true_values <= cal_range_max) & ...
                (predicted_values >= cal_range_min & predicted_values <= cal_range_max);
    n_range = sum(range_idx);
    
    fprintf('Found %d samples in [%.3f, %.3f] range...\n', n_range, cal_range_min, cal_range_max);
    
    if n_range < 10
        warning('Limited calibration data (%d samples) in [%.3f, %.3f] range', ...
                n_range, cal_range_min, cal_range_max);
        
        % Fallback: expand range by 50%
        range_expansion = (cal_range_max - cal_range_min) * 0.5;
        fallback_min = max(0, cal_range_min - range_expansion);
        fallback_max = cal_range_max + range_expansion;
        
        fallback_idx = (true_values >= fallback_min & true_values <= fallback_max) & ...
                       (predicted_values >= fallback_min & predicted_values <= fallback_max);
        
        if sum(fallback_idx) >= 10
            range_idx = fallback_idx;
            n_range = sum(range_idx);
            fprintf('Using expanded range [%.3f, %.3f]: %d samples\n', ...
                    fallback_min, fallback_max, n_range);
            cal_range_min = fallback_min;
            cal_range_max = fallback_max;
        else
            % No calibration if insufficient data
            calibration_params = struct('slope', 1, 'intercept', 0, 'n_samples', 0, ...
                                      'range_min', cal_range_min, 'range_max', cal_range_max, ...
                                      'r2', NaN, 'rmse', NaN);
            warning('Insufficient data for calibration - using identity transformation');
            return;
        end
    end
    
    % Extract range data
    y_true_range = true_values(range_idx);
    y_pred_range = predicted_values(range_idx);
    
    % Fit linear relationship: true = slope * predicted + intercept
    p = polyfit(y_pred_range, y_true_range, 1);
    slope = p(1);
    intercept = p(2);
    
    % Calculate calibration quality metrics
    y_cal_pred = slope * y_pred_range + intercept;
    cal_r2 = 1 - sum((y_true_range - y_cal_pred).^2) / sum((y_true_range - mean(y_true_range)).^2);
    cal_rmse = sqrt(mean((y_cal_pred - y_true_range).^2));
    
    % Store calibration parameters
    calibration_params = struct();
    calibration_params.slope = slope;
    calibration_params.intercept = intercept;
    calibration_params.n_samples = n_range;
    calibration_params.r2 = cal_r2;
    calibration_params.rmse = cal_rmse;
    calibration_params.range_min = cal_range_min;
    calibration_params.range_max = cal_range_max;
    
    fprintf('Calibration learned: true = %.4f * predicted + %.6f (range: [%.3f, %.3f])\n', ...
            slope, intercept, cal_range_min, cal_range_max);
    fprintf(' Calibration fit: R² = %.4f, RMSE = %.6f (n = %d)\n', cal_r2, cal_rmse, n_range);
    
    % Interpretation
    if slope < 1
        fprintf(' Interpretation: Predictions in [%.3f, %.3f] range will be reduced by %.1f%%\n', ...
                cal_range_min, cal_range_max, (1-slope)*100);
    elseif slope > 1
        fprintf(' Interpretation: Predictions in [%.3f, %.3f] range will be increased by %.1f%%\n', ...
                cal_range_min, cal_range_max, (slope-1)*100);
    else
        fprintf(' Interpretation: Predictions in [%.3f, %.3f] range will remain unchanged\n', ...
                cal_range_min, cal_range_max);
    end
    
    % Show range coverage
    range_width = cal_range_max - cal_range_min;
    fprintf(' Range coverage: %.3f width, %.1f%% of [0,1] space\n', range_width, range_width*100);
end

function calibrated_p = applyCalibration(raw_predictions, calibration_params)
    % Apply calibration correction ONLY to predictions in the calibration range
    
    if calibration_params.n_samples == 0
        % No calibration learned - return original predictions
        calibrated_p = raw_predictions;
        return;
    end
    
    % Initialize with original predictions
    calibrated_p = raw_predictions;
    
    % Only calibrate predictions within the learned range
    cal_range_idx = raw_predictions <= calibration_params.range_max;
    n_to_calibrate = sum(cal_range_idx);
    
    if n_to_calibrate > 0
        % Apply linear rescaling only to low p-values
        calibrated_subset = calibration_params.slope * raw_predictions(cal_range_idx) + calibration_params.intercept;
        
        % Ensure valid p-values for calibrated subset
        calibrated_subset = max(0, min(1, calibrated_subset));
        
        % Update only the calibrated portion
        calibrated_p(cal_range_idx) = calibrated_subset;
        
        fprintf('Applied calibration to %d predictions in 0-%.3f range\n', n_to_calibrate, calibration_params.range_max);
        
        % Report any clipping
        original_cal_values = calibration_params.slope * raw_predictions(cal_range_idx) + calibration_params.intercept;
        n_clipped_low = sum(calibrated_subset == 0 & original_cal_values < 0);
        n_clipped_high = sum(calibrated_subset == 1 & original_cal_values > 1);
        
        if n_clipped_low > 0 || n_clipped_high > 0
            fprintf('  Calibration clipping: %d clipped to 0, %d clipped to 1\n', n_clipped_low, n_clipped_high);
        end
    else
        fprintf('No predictions in calibration range (0-%.3f) - no calibration applied\n', calibration_params.range_max);
    end
end

function data_oversampled = oversamplePValueRange(data, target_range_min, target_range_max, oversample_factor, verbose)
    % Oversample data within a specific p-value range to improve model training
    % 
    % INPUTS:
    %   data - table with BiasedP, UnbiasedP, and other columns
    %   target_range_min - minimum p-value for oversampling range (default: 0.04)
    %   target_range_max - maximum p-value for oversampling range (default: 0.06)
    %   oversample_factor - multiplicative factor for oversampling (default: 3.0)
    %   verbose - whether to print detailed information (default: true)
    %
    % OUTPUT:
    %   data_oversampled - table with additional samples in target range
    
    % Set defaults
    if nargin < 2 || isempty(target_range_min)
        target_range_min = 0.04;
    end
    if nargin < 3 || isempty(target_range_max)
        target_range_max = 0.06;
    end
    if nargin < 4 || isempty(oversample_factor)
        oversample_factor = 3.0;
    end
    if nargin < 5 || isempty(verbose)
        verbose = true;
    end
    
    % Validate inputs
    if target_range_min >= target_range_max
        error('target_range_min (%.4f) must be less than target_range_max (%.4f)', ...
              target_range_min, target_range_max);
    end
    if oversample_factor < 1.0
        error('oversample_factor (%.2f) must be >= 1.0', oversample_factor);
    end
    
    if verbose
        fprintf('\n=== OVERSAMPLING P-VALUE RANGE [%.4f, %.4f] ===\n', ...
                target_range_min, target_range_max);
        fprintf('Oversample factor: %.2fx\n', oversample_factor);
    end
    
    % Find samples in target range - check BOTH BiasedP AND UnbiasedP
    % This ensures we oversample the most relevant samples for model training
    target_range_idx = (data.BiasedP >= target_range_min & data.BiasedP <= target_range_max) | ...
                       (data.UnbiasedP >= target_range_min & data.UnbiasedP <= target_range_max);
    
    n_target_samples = sum(target_range_idx);
    n_total_original = height(data);
    
    if verbose
        fprintf('Found %d samples in target range (%.2f%% of original data)\n', ...
                n_target_samples, 100*n_target_samples/n_total_original);
    end
    
    if n_target_samples == 0
        warning('No samples found in target range [%.4f, %.4f] - no oversampling applied', ...
                target_range_min, target_range_max);
        data_oversampled = data;
        return;
    end
    
    % Extract target range data
    target_data = data(target_range_idx, :);
    
    % Calculate how many additional samples we need
    n_additional = round((oversample_factor - 1) * n_target_samples);
    
    if verbose
        fprintf('Generating %d additional samples (%.2fx - 1 = %.2fx additional)\n', ...
                n_additional, oversample_factor, oversample_factor - 1);
    end
    
    if n_additional <= 0
        if verbose
            fprintf('Oversample factor too small - no additional samples generated\n');
        end
        data_oversampled = data;
        return;
    end
    
    % Generate additional samples through bootstrap resampling WITH SLIGHT NOISE
    % This prevents exact duplicates while maintaining the distribution characteristics
    rng('shuffle'); % Use current time for randomness
    
    % Bootstrap indices
    bootstrap_idx = randi(n_target_samples, n_additional, 1);
    additional_data = target_data(bootstrap_idx, :);
    
    % Add small amount of Gaussian noise to continuous variables to create variation
    % Noise level is proportional to the standard deviation of each variable
    noise_factor = 0.01; % 1% noise relative to std dev
    
    continuous_vars = {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', 'Skewness', 'Kurtosis'};
    for i = 1:length(continuous_vars)
        var_name = continuous_vars{i};
        if ismember(var_name, data.Properties.VariableNames)
            original_values = additional_data.(var_name);
            var_std = std(target_data.(var_name));
            
            % Add noise but ensure values stay within reasonable bounds
            noise = randn(size(original_values)) * var_std * noise_factor;
            noisy_values = original_values + noise;
            
            % Apply bounds based on variable type
            switch var_name
                case {'BiasedP', 'UnbiasedP'}
                    % P-values must be in [0, 1]
                    noisy_values = max(0, min(1, noisy_values));
                case 'Variance'
                    % Variance must be non-negative
                    noisy_values = max(0, noisy_values);
                % Mean, Skewness, Kurtosis can be any real number - no bounds needed
            end
            
            additional_data.(var_name) = noisy_values;
        end
    end
    
    % Combine original data with additional samples
    data_oversampled = [data; additional_data];
    
    if verbose
        fprintf('\nOVERSAMPLING RESULTS:\n');
        fprintf('Original dataset size: %d samples\n', n_total_original);
        fprintf('Oversampled dataset size: %d samples\n', height(data_oversampled));
        fprintf('Total increase: %d samples (%.1fx larger)\n', ...
                height(data_oversampled) - n_total_original, ...
                height(data_oversampled) / n_total_original);
        
        % Show distribution of samples in target range
        new_target_count = sum((data_oversampled.BiasedP >= target_range_min & data_oversampled.BiasedP <= target_range_max) | ...
                              (data_oversampled.UnbiasedP >= target_range_min & data_oversampled.UnbiasedP <= target_range_max));
        fprintf('Samples in target range after oversampling: %d (%.1f%% of final dataset)\n', ...
                new_target_count, 100*new_target_count/height(data_oversampled));
        
        % Verify the actual oversampling achieved
        actual_oversample_factor = new_target_count / n_target_samples;
        fprintf('Actual oversampling factor for target range: %.2fx\n', actual_oversample_factor);
    end
    
    % Shuffle the final dataset to mix original and oversampled data
    shuffled_idx = randperm(height(data_oversampled));
    data_oversampled = data_oversampled(shuffled_idx, :);
    
    if verbose
        fprintf('Final dataset shuffled to mix original and oversampled data\n');
        fprintf('=== OVERSAMPLING COMPLETE ===\n');
    end
end