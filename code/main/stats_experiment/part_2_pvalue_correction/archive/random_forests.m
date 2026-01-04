% Random Forest Model for Permutation Test P-values
% CPU-optimized version with fast data generation
% Supports PARQUET format for faster I/O
% FLEXIBLE: Choose between one-sample or paired two-sample tests
% MODIFIED: Flexible filtering - exclude selected distributions AND high skewness samples
% NEW: Option to use theoretical moments instead of sample-level moments
% UPDATED: Random Forest implementation replacing polynomial regression
% NEW: Range-specific training - train separate models on different BiasedP ranges

clear; close all; clc;

%% Parameters
nPValues = 50000000;
nParticipants = 150;
nPermutations = 3000;

TEST_TYPE = 'one-sample';  % 'one-sample' or 'paired'

% FLEXIBLE FILTERING CONFIGURATION
DISTRIBUTIONS_TO_EXCLUDE = {};
DISTRIBUTIONS_TO_EXCLUDE = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T', 'EXPONENTIAL', 'LOGNORMAL'};
%DISTRIBUTIONS_TO_EXCLUDE = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T'};
DISTRIBUTIONS_TO_EXCLUDE = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T', 'EXPONENTIAL', 'LOGNORMAL'};
SKEWNESS_THRESHOLD = Inf;

USE_CALIBRATION = false;
USE_THEORETICAL_MOMENTS = false;

% RANDOM FOREST PARAMETERS
RF_NUM_TREES = 100;
RF_MIN_LEAF_SIZE = 2;
RF_MTRY = 3;

% TRAINING DATA SAMPLING
TRAINING_SAMPLE_FRACTION = 0.1;  % <-- CHANGE THIS: Fraction of training data to use (0.01 to 1.0)

% OVERSAMPLING CONFIGURATION
USE_OVERSAMPLING = true;
OVERSAMPLE_RANGE_MIN = 0.001;
OVERSAMPLE_RANGE_MAX = 0.04;
OVERSAMPLE_FACTOR = 3.0;

%% === NEW: RANGE-SPECIFIC TRAINING CONFIGURATION ===
USE_RANGE_SPECIFIC_TRAINING = true;  % Set to false for single-model training

% Define training ranges: Each row is [min_BiasedP, max_BiasedP]
% Only training data within these ranges will be used to train each model
% Examples:
%   Single focused range: {[0.04, 0.06]}
%   Multiple ranges: {[0, 0.04], [0.04, 0.1], [0.1, 1.0]}
%   Full range (equivalent to traditional): {[0, 1.0]}

TRAINING_RANGES = {[0, 0.04], [0.04, 0.06], [0.06, 1]};  % <-- CUSTOMIZE THIS


TRAINING_RANGES = {[0, 1]};  % <-- CUSTOMIZE THIS

% Prediction strategy for test samples whose BiasedP falls outside all ranges:
%   'nearest' - use the model from the nearest range boundary
%   'passthrough' - return original BiasedP (no correction)
OUT_OF_RANGE_STRATEGY = 'nearest';  % <-- CHOOSE STRATEGY
%% ===============================================

%% Set random seed for reproducibility
rng(42);

%% Load existing data or generate new
load_existing = true;

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
    
    % Apply filtering
    fprintf('\n=== APPLYING FLEXIBLE FILTERING ===\n');
    before_filtering = height(data);
    keep_idx = true(height(data), 1);
    
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
    
    if isfinite(SKEWNESS_THRESHOLD)
        skewness_filter = abs(data.Skewness) >= SKEWNESS_THRESHOLD;
        keep_idx = keep_idx & ~skewness_filter;
    else
        fprintf('Skewness threshold disabled (set to Inf).\n');
    end
    
    data = data(keep_idx, :);
    
    fprintf('\nFINAL FILTERING RESULTS:\n');
    fprintf('Original samples: %d\n', before_filtering);
    fprintf('Final samples: %d (%.1f%% remaining)\n', height(data), 100*height(data)/before_filtering);
    
    % Create filter description
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
    
    % Show remaining distributions
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
    % Generate new data
    fprintf('Generating training data using FAST %s permutation tests...\n', upper(TEST_TYPE));
    fprintf('Target: %d p-values, %d participants per group, %d permutations\n', ...
            nPValues, nParticipants, nPermutations);
    
    tic;
    data = generatePermutationDataFast(nPValues, nParticipants, nPermutations, TEST_TYPE);
    elapsed = toc;
    
    fprintf('Generated %d samples in %.2f seconds (%.1f samples/sec)\n', ...
            height(data), elapsed, height(data)/elapsed);
    
    % Apply filtering
    fprintf('\nApplying flexible filters...\n');
    original_size = height(data);
    keep_idx = true(height(data), 1);
    
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
    
    if isfinite(SKEWNESS_THRESHOLD)
        skewness_filter = abs(data.Skewness) >= SKEWNESS_THRESHOLD;
        keep_idx = keep_idx & ~skewness_filter;
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
            height(data), 100*height(data)/original_size);
end

%% Apply theoretical moments if requested
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

% Display summary statistics
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

fprintf('%-15s %8d %10.4f %10.4f %10.4f %10.4f\n', ...
        'OVERALL', height(data), ...
        mean(data.Skewness), std(data.Skewness), ...
        mean(data.Kurtosis), std(data.Kurtosis));

%% Perform Train/Test Split and Model Training
if USE_RANGE_SPECIFIC_TRAINING
    fprintf('\n=== RANGE-SPECIFIC TRAINING ENABLED ===\n');
    fprintf('Number of ranges: %d\n', length(TRAINING_RANGES));
    for i = 1:length(TRAINING_RANGES)
        range = TRAINING_RANGES{i};
        fprintf('  Range %d: [%.4f, %.4f]\n', i, range(1), range(2));
    end
    fprintf('Out-of-range strategy: %s\n', OUT_OF_RANGE_STRATEGY);
    fprintf('Performing stratified 80/20 train/test split with range filtering...\n');
    fprintf('Training Random Forest models...\n');
    
    [results, models] = trainTestSplitRangeSpecific(data, TRAINING_RANGES, OUT_OF_RANGE_STRATEGY, ...
                                                     USE_CALIBRATION, RF_NUM_TREES, RF_MIN_LEAF_SIZE, RF_MTRY, ...
                                                     TRAINING_SAMPLE_FRACTION);
else
    fprintf('\nPerforming stratified 80/20 train/test split (%s)...\n', filter_description);
    fprintf('Training Random Forest model with %d trees...\n', RF_NUM_TREES);
    
    [results, models] = trainTestSplit(data, USE_CALIBRATION, RF_NUM_TREES, RF_MIN_LEAF_SIZE, RF_MTRY, ...
                                       TRAINING_SAMPLE_FRACTION);
end

%% Visualize Results
fprintf('\nVisualizing results...\n');

moments_type = ternary(USE_THEORETICAL_MOMENTS, 'Theoretical', 'Sample');
if USE_CALIBRATION && results.use_calibration && ~isempty(results.calibration_params)
    model_title = sprintf('Random Forest (%d trees): ', RF_NUM_TREES);
    results_for_viz = results;
    results_for_viz.test_pred = results.test_pred_cal;
    results_for_viz.test_critical_idx = results.test_critical_idx_cal;
    results_for_viz.test_rmse = results.test_rmse_cal;
    results_for_viz.test_r2 = results.test_r2_cal;
    visualizeResults(data, results_for_viz, models, model_title);
else
    model_title = sprintf('Random Forest (%d trees): ', RF_NUM_TREES);
    visualizeResults(data, results, models, model_title);
end

%% Final cleanup
close all;
clearvars;

%% ============================================================================
%% RANGE-SPECIFIC TRAINING FUNCTIONS
%% ============================================================================

function [results, models] = trainTestSplitRangeSpecific(data, training_ranges, out_of_range_strategy, ...
                                                          use_calibration, num_trees, min_leaf_size, mtry, ...
                                                          training_sample_fraction)
    % Train separate Random Forest models for different BiasedP ranges
    
    if nargin < 8
        training_sample_fraction = 0.1;
    end
    
    n_ranges = length(training_ranges);
    
    % Validate ranges
    for i = 1:n_ranges
        range = training_ranges{i};
        if length(range) ~= 2 || range(1) >= range(2)
            error('Invalid range %d: must be [min, max] with min < max', i);
        end
        if range(1) < 0 || range(2) > 1
            error('Invalid range %d: BiasedP must be in [0, 1]', i);
        end
    end
    
    % Perform stratified train/test split
    unique_distributions = unique(data.Distribution);
    nDistributions = length(unique_distributions);
    
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
    
    % Stratified split
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_indices = find(strcmp(data.Distribution, dist_name));
        n_samples = length(dist_indices);
        
        perm_indices = dist_indices(randperm(n_samples));
        
        n_train = round(train_pct * n_samples);
        if use_calibration
            n_cal = round(cal_pct * n_samples);
            n_test = n_samples - n_train - n_cal;
        else
            n_cal = 0;
            n_test = n_samples - n_train;
        end
        
        train_idx(perm_indices(1:n_train)) = true;
        if use_calibration && n_cal > 0
            cal_idx(perm_indices(n_train+1:n_train+n_cal)) = true;
            test_idx(perm_indices(n_train+n_cal+1:end)) = true;
        else
            test_idx(perm_indices(n_train+1:end)) = true;
        end
    end
    
    train_data = data(train_idx, :);
    test_data = data(test_idx, :);
    if use_calibration
        cal_data = data(cal_idx, :);
    else
        cal_data = [];
    end
    
    % Train separate model for each range
    models = cell(n_ranges, 1);
    train_samples_per_range = zeros(n_ranges, 1);
    
    for r = 1:n_ranges
        range = training_ranges{r};
        range_min = range(1);
        range_max = range(2);
        
        fprintf('\n=== Training Range %d: [%.4f, %.4f] ===\n', r, range_min, range_max);
        
        % Filter training data by BiasedP range
        range_train_idx = (train_data.BiasedP >= range_min) & (train_data.BiasedP <= range_max);
        train_data_range = train_data(range_train_idx, :);
        train_samples_per_range(r) = height(train_data_range);
        
        fprintf('Training samples in range: %d (%.1f%% of training set)\n', ...
                height(train_data_range), 100*height(train_data_range)/height(train_data));
        
        if height(train_data_range) < 100
            warning('Very few training samples (%d) in range [%.4f, %.4f]', ...
                    height(train_data_range), range_min, range_max);
        end
        
        % Take sample for faster training
        n_sample = max(100, round(training_sample_fraction * height(train_data_range)));
        n_sample = min(n_sample, height(train_data_range));
        sample_idx = randperm(height(train_data_range), n_sample);
        train_data_sample = train_data_range(sample_idx, :);
        
        % Create feature matrix
        X_train = [train_data_sample.BiasedP, train_data_sample.Mean, ...
                   train_data_sample.Skewness, train_data_sample.Variance, ...
                   train_data_sample.Kurtosis];
        y_train = train_data_sample.UnbiasedP;
        
        fprintf('Using %d samples (%.1f%% sample) for training range %d\n', ...
                n_sample, 100*training_sample_fraction, r);
        
        % Train Random Forest
        fprintf('Training Random Forest: %d trees, min_leaf=%d, mtry=%d...\n', ...
                num_trees, min_leaf_size, mtry);
        tic;
        model = TreeBagger(num_trees, X_train, y_train, ...
                          'Method', 'regression', ...
                          'OOBPrediction', 'on', ...
                          'MinLeafSize', min_leaf_size, ...
                          'NumPredictorsToSample', mtry, ...
                          'Options', statset('Display', 'off', 'UseParallel', true));
        train_time = toc;
        fprintf('Training completed in %.2f seconds\n', train_time);
        fprintf('Out-of-Bag Error: %.6f\n', oobError(model, 'Mode', 'Ensemble'));
        
        models{r} = model;
    end
    
    % Make predictions on test set (vectorized by range)
    fprintf('\n=== Making Predictions on Test Set ===\n');
    n_test = height(test_data);
    y_test_pred_raw = zeros(n_test, 1);
    model_assignment = zeros(n_test, 1);
    
    % Process each range - vectorized predictions
    for r = 1:n_ranges
        range = training_ranges{r};
        range_min = range(1);
        range_max = range(2);
        
        % Find all test samples in this range
        in_range_idx = (test_data.BiasedP >= range_min) & (test_data.BiasedP <= range_max);
        n_in_range = sum(in_range_idx);
        
        if n_in_range > 0
            % Vectorized prediction for all samples in this range at once
            X_test_range = [test_data.BiasedP(in_range_idx), test_data.Mean(in_range_idx), ...
                           test_data.Skewness(in_range_idx), test_data.Variance(in_range_idx), ...
                           test_data.Kurtosis(in_range_idx)];
            y_test_pred_raw(in_range_idx) = predictRandomForest(models{r}, X_test_range);
            model_assignment(in_range_idx) = r;
            fprintf('  Range %d [%.4f, %.4f]: %d predictions\n', r, range_min, range_max, n_in_range);
        end
    end
    
    % Handle out-of-range samples
    out_of_range_idx = (model_assignment == 0);
    n_out_of_range = sum(out_of_range_idx);
    
    if n_out_of_range > 0
        fprintf('Processing %d out-of-range samples using %s strategy...\n', ...
                n_out_of_range, out_of_range_strategy);
        
        switch out_of_range_strategy
            case 'nearest'
                % Find nearest range for each out-of-range sample (vectorized)
                out_of_range_biased_p = test_data.BiasedP(out_of_range_idx);
                n_oor = length(out_of_range_biased_p);
                
                % Calculate distance to each range for all out-of-range samples
                min_distances = inf(n_oor, 1);
                nearest_range_idx = ones(n_oor, 1);
                
                for r = 1:n_ranges
                    range = training_ranges{r};
                    
                    % Distance to this range (0 if inside, otherwise min distance to boundaries)
                    distances = zeros(n_oor, 1);
                    below = out_of_range_biased_p < range(1);
                    above = out_of_range_biased_p > range(2);
                    
                    distances(below) = range(1) - out_of_range_biased_p(below);
                    distances(above) = out_of_range_biased_p(above) - range(2);
                    
                    % Update nearest range
                    closer = distances < min_distances;
                    min_distances(closer) = distances(closer);
                    nearest_range_idx(closer) = r;
                end
                
                % Make predictions using nearest models (grouped by range for efficiency)
                for r = 1:n_ranges
                    use_this_model = (nearest_range_idx == r);
                    if any(use_this_model)
                        % Find indices in original test_data
                        oor_indices = find(out_of_range_idx);
                        global_indices = oor_indices(use_this_model);
                        
                        X_test_nearest = [test_data.BiasedP(global_indices), ...
                                        test_data.Mean(global_indices), ...
                                        test_data.Skewness(global_indices), ...
                                        test_data.Variance(global_indices), ...
                                        test_data.Kurtosis(global_indices)];
                        y_test_pred_raw(global_indices) = predictRandomForest(models{r}, X_test_nearest);
                        model_assignment(global_indices) = r;
                    end
                end
                
            case 'passthrough'
                % Simply use BiasedP as prediction for out-of-range samples
                y_test_pred_raw(out_of_range_idx) = test_data.BiasedP(out_of_range_idx);
                model_assignment(out_of_range_idx) = 0;
        end
    end
    
    % Show prediction statistics
    fprintf('\nTest set prediction statistics:\n');
    for r = 1:n_ranges
        range = training_ranges{r};
        n_assigned = sum(model_assignment == r);
        fprintf('  Range %d [%.4f, %.4f]: %d predictions (%.1f%% of test set)\n', ...
                r, range(1), range(2), n_assigned, 100*n_assigned/length(model_assignment));
    end
    
    if strcmp(out_of_range_strategy, 'nearest')
        n_out_of_range = sum(model_assignment == 0);
        if n_out_of_range > 0
            fprintf('  Out-of-range (using nearest): %d predictions (%.1f%% of test set)\n', ...
                    n_out_of_range, 100*n_out_of_range/length(model_assignment));
        end
    end
    
    % Calibration (if enabled)
    calibration_params = [];
    if use_calibration && ~isempty(cal_data)
        fprintf('\n=== Learning Calibration Per Range ===\n');
        calibration_params = cell(n_ranges, 1);
        
        for r = 1:n_ranges
            range = training_ranges{r};
            fprintf('Calibrating range %d: [%.4f, %.4f]\n', r, range(1), range(2));
            
            % Filter calibration data by range
            range_cal_idx = (cal_data.BiasedP >= range(1)) & (cal_data.BiasedP <= range(2));
            cal_data_range = cal_data(range_cal_idx, :);
            
            if height(cal_data_range) >= 10
                X_cal = [cal_data_range.BiasedP, cal_data_range.Mean, ...
                         cal_data_range.Skewness, cal_data_range.Variance, ...
                         cal_data_range.Kurtosis];
                y_cal = cal_data_range.UnbiasedP;
                y_cal_pred = predictRandomForest(models{r}, X_cal);
                
                calibration_params{r} = learnCalibration(y_cal, y_cal_pred);
            else
                fprintf('  Insufficient calibration data for range %d (%d samples)\n', ...
                        r, height(cal_data_range));
                calibration_params{r} = struct('slope', 1, 'intercept', 0, 'n_samples', 0);
            end
        end
    end
    
    % Apply calibration
    if use_calibration && ~isempty(calibration_params)
        fprintf('\n=== Applying Calibration to Test Set ===\n');
        y_test_pred_cal = y_test_pred_raw;
        
        for r = 1:n_ranges
            range_test_idx = (model_assignment == r);
            if sum(range_test_idx) > 0 && calibration_params{r}.n_samples > 0
                y_test_pred_cal(range_test_idx) = applyCalibration(y_test_pred_raw(range_test_idx), ...
                                                                    calibration_params{r});
            end
        end
    else
        y_test_pred_cal = y_test_pred_raw;
    end
    
    % Calculate metrics
    y_test = test_data.UnbiasedP;
    
    % Uncalibrated metrics
    test_critical_idx_raw = (y_test <= 0.05) & (y_test_pred_raw <= 0.05);
    if sum(test_critical_idx_raw) > 0
        test_rmse_raw = sqrt(mean((y_test_pred_raw(test_critical_idx_raw) - y_test(test_critical_idx_raw)).^2));
        test_r2_raw = 1 - sum((y_test(test_critical_idx_raw) - y_test_pred_raw(test_critical_idx_raw)).^2) / ...
                      sum((y_test(test_critical_idx_raw) - mean(y_test(test_critical_idx_raw))).^2);
        test_mae_raw = mean(abs(y_test_pred_raw(test_critical_idx_raw) - y_test(test_critical_idx_raw)));
    else
        test_rmse_raw = NaN; test_r2_raw = NaN; test_mae_raw = NaN;
    end
    test_fpr_raw = sum(y_test_pred_raw <= 0.05) / length(y_test_pred_raw);
    
    % Calibrated metrics
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
    results.test_pred_raw = y_test_pred_raw;
    results.test_pred_cal = y_test_pred_cal;
    results.test_critical_idx_raw = test_critical_idx_raw;
    results.test_critical_idx_cal = test_critical_idx_cal;
    results.calibration_params = calibration_params;
    results.use_calibration = use_calibration;
    results.model_assignment = model_assignment;
    results.training_ranges = training_ranges;
    results.train_samples_per_range = train_samples_per_range;
    
    results.test_rmse_raw = test_rmse_raw;
    results.test_r2_raw = test_r2_raw;
    results.test_mae_raw = test_mae_raw;
    results.test_fpr_raw = test_fpr_raw;
    results.test_rmse_cal = test_rmse_cal;
    results.test_r2_cal = test_r2_cal;
    results.test_mae_cal = test_mae_cal;
    results.test_fpr_cal = test_fpr_cal;
    
    % Print results
    fprintf('\n========== RANGE-SPECIFIC RANDOM FOREST PERFORMANCE ==========\n');
    fprintf('Number of ranges: %d\n', n_ranges);
    fprintf('Training samples per range:\n');
    for r = 1:n_ranges
        range = training_ranges{r};
        fprintf('  Range %d [%.4f, %.4f]: %d samples\n', r, range(1), range(2), train_samples_per_range(r));
    end
    
    fprintf('\nTest Set - UNCALIBRATED (n=%d critical samples):\n', sum(test_critical_idx_raw));
    fprintf('  RMSE: %.6f, R²: %.4f, MAE: %.6f, FPR: %.4f\n', test_rmse_raw, test_r2_raw, test_mae_raw, test_fpr_raw);
    
    if use_calibration && ~isempty(calibration_params)
        fprintf('Test Set - CALIBRATED (n=%d critical samples):\n', sum(test_critical_idx_cal));
        fprintf('  RMSE: %.6f, R²: %.4f, MAE: %.6f, FPR: %.4f\n', test_rmse_cal, test_r2_cal, test_mae_cal, test_fpr_cal);
    end
    fprintf('==============================================================\n');
end

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

function data = loadParquetData(filename)
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

function data_modified = replaceWithTheoreticalMoments(data)
    data_modified = data;
    unique_distributions = unique(data.Distribution);
    
    fprintf('Processing %d distributions...\n', length(unique_distributions));
    
    for i = 1:length(unique_distributions)
        dist_name = unique_distributions{i};
        dist_idx = strcmp(data.Distribution, dist_name);
        n_samples = sum(dist_idx);
        
        if n_samples == 0
            continue;
        end
        
        theoretical_moments = getTheoreticalMoments(dist_name);
        
        data_modified.Mean(dist_idx) = theoretical_moments(1);
        data_modified.Variance(dist_idx) = theoretical_moments(2);
        data_modified.Skewness(dist_idx) = theoretical_moments(3);
        data_modified.Kurtosis(dist_idx) = theoretical_moments(4);
        
        fprintf('  %-12s: %d samples | Theoretical moments = [μ=%.4f, σ²=%.4f, γ₁=%.4f, γ₂=%.4f]\n', ...
                dist_name, n_samples, theoretical_moments(1), theoretical_moments(2), ...
                theoretical_moments(3), theoretical_moments(4));
    end
    
    fprintf('Replacement complete!\n');
end

function moments = getTheoreticalMoments(distribution_name)
    switch upper(distribution_name)
        case 'NORMAL'
            mean_val = 0;
            variance = 1;
            skewness = 0;
            excess_kurtosis = 0;
            
        case 'UNIFORM'
            mean_val = 0;
            variance = (0.5 - (-0.5))^2 / 12;
            skewness = 0;
            excess_kurtosis = -1.2;
            
        case 'LAPLACE'
            mean_val = 0;
            variance = 2;
            skewness = 0;
            excess_kurtosis = 3;
            
        case 'STUDENT_T'
            df = 3;
            mean_val = 0;
            if df > 2
                variance = df / (df - 2);
            else
                variance = Inf;
            end
            skewness = 0;
            if df > 4
                excess_kurtosis = 6 / (df - 4);
            else
                excess_kurtosis = 6;
            end
            
        case 'GAMMA'
            shape = 0.5; scale = 1;
            mean_val = 0;
            variance = shape * scale^2;
            skewness = 2 / sqrt(shape);
            excess_kurtosis = 6 / shape;
            
        case 'EXPONENTIAL'
            rate = 1;
            mean_val = 0;
            variance = 1 / rate^2;
            skewness = 2;
            excess_kurtosis = 6;
            
        case 'LOGNORMAL'
            mu = 0; sigma = 0.5;
            mean_val = 0;
            variance = (exp(sigma^2) - 1) * exp(2*mu + sigma^2);
            skewness = (exp(sigma^2) + 2) * sqrt(exp(sigma^2) - 1);
            excess_kurtosis = exp(4*sigma^2) + 2*exp(3*sigma^2) + 3*exp(2*sigma^2) - 6;
            
        otherwise
            error('Unknown distribution: %s', distribution_name);
    end
    
    moments = [mean_val, variance, skewness, excess_kurtosis];
end

function data_oversampled = oversamplePValueRange(data, target_range_min, target_range_max, oversample_factor, verbose)
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
    
    if verbose
        fprintf('\n=== OVERSAMPLING P-VALUE RANGE [%.4f, %.4f] ===\n', ...
                target_range_min, target_range_max);
        fprintf('Oversample factor: %.2fx\n', oversample_factor);
    end
    
    target_range_idx = (data.BiasedP >= target_range_min & data.BiasedP <= target_range_max) | ...
                       (data.UnbiasedP >= target_range_min & data.UnbiasedP <= target_range_max);
    
    n_target_samples = sum(target_range_idx);
    n_total_original = height(data);
    
    if verbose
        fprintf('Found %d samples in target range (%.2f%% of original data)\n', ...
                n_target_samples, 100*n_target_samples/n_total_original);
    end
    
    if n_target_samples == 0
        warning('No samples found in target range - no oversampling applied');
        data_oversampled = data;
        return;
    end
    
    target_data = data(target_range_idx, :);
    n_additional = round((oversample_factor - 1) * n_target_samples);
    
    if verbose
        fprintf('Generating %d additional samples\n', n_additional);
    end
    
    if n_additional <= 0
        data_oversampled = data;
        return;
    end
    
    rng('shuffle');
    bootstrap_idx = randi(n_target_samples, n_additional, 1);
    additional_data = target_data(bootstrap_idx, :);
    
    noise_factor = 0.01;
    continuous_vars = {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', 'Skewness', 'Kurtosis'};
    for i = 1:length(continuous_vars)
        var_name = continuous_vars{i};
        if ismember(var_name, data.Properties.VariableNames)
            original_values = additional_data.(var_name);
            var_std = std(target_data.(var_name));
            
            noise = randn(size(original_values)) * var_std * noise_factor;
            noisy_values = original_values + noise;
            
            switch var_name
                case {'BiasedP', 'UnbiasedP'}
                    noisy_values = max(0, min(1, noisy_values));
                case 'Variance'
                    noisy_values = max(0, noisy_values);
            end
            
            additional_data.(var_name) = noisy_values;
        end
    end
    
    data_oversampled = [data; additional_data];
    
    if verbose
        fprintf('\nOVERSAMPLING RESULTS:\n');
        fprintf('Original dataset size: %d samples\n', n_total_original);
        fprintf('Oversampled dataset size: %d samples\n', height(data_oversampled));
        fprintf('Total increase: %d samples (%.1fx larger)\n', ...
                height(data_oversampled) - n_total_original, ...
                height(data_oversampled) / n_total_original);
    end
    
    shuffled_idx = randperm(height(data_oversampled));
    data_oversampled = data_oversampled(shuffled_idx, :);
end

function data = generatePermutationDataFast(nPValues, nParticipants, nPermutations, test_type)
    % Note: This function requires the full implementation from your original script
    % Including: permutation test functions, distribution generation, etc.
    % For brevity, this is a placeholder - copy from your original script
    error('generatePermutationDataFast not included - copy from original script');
end

function [results, model] = trainTestSplit(data, use_calibration, num_trees, min_leaf_size, mtry, training_sample_fraction)
    % Original single-model training (from your script)
    
    if nargin < 2
        use_calibration = false;
    end
    if nargin < 3
        num_trees = 100;
    end
    if nargin < 4
        min_leaf_size = 5;
    end
    if nargin < 5
        mtry = 2;
    end
    if nargin < 6
        training_sample_fraction = 0.1;
    end
    
    unique_distributions = unique(data.Distribution);
    nDistributions = length(unique_distributions);
    
    train_idx = false(height(data), 1);
    cal_idx = false(height(data), 1);
    test_idx = false(height(data), 1);
    
    if use_calibration
        train_pct = 0.6;
        cal_pct = 0.2;
        test_pct = 0.2;
    else
        train_pct = 0.8;
        cal_pct = 0.0;
        test_pct = 0.2;
    end
    
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_indices = find(strcmp(data.Distribution, dist_name));
        n_samples = length(dist_indices);
        
        perm_indices = dist_indices(randperm(n_samples));
        
        n_train = round(train_pct * n_samples);
        if use_calibration
            n_cal = round(cal_pct * n_samples);
            n_test = n_samples - n_train - n_cal;
        else
            n_cal = 0;
            n_test = n_samples - n_train;
        end
        
        train_idx(perm_indices(1:n_train)) = true;
        if use_calibration && n_cal > 0
            cal_idx(perm_indices(n_train+1:n_train+n_cal)) = true;
            test_idx(perm_indices(n_train+n_cal+1:end)) = true;
        else
            test_idx(perm_indices(n_train+1:end)) = true;
        end
    end
    
    train_data = data(train_idx, :);
    test_data = data(test_idx, :);
    if use_calibration
        cal_data = data(cal_idx, :);
    else
        cal_data = [];
    end
    
    n_sample = round(training_sample_fraction * height(train_data));
    sample_idx = randperm(height(train_data), n_sample);
    train_data_sample = train_data(sample_idx, :);
    
    X_train = [train_data_sample.BiasedP, train_data_sample.Mean, train_data_sample.Skewness, ...
               train_data_sample.Variance, train_data_sample.Kurtosis];
    y_train = train_data_sample.UnbiasedP;
    
    fprintf('Using %d samples (%.1f%% of training data) for Random Forest training\n', ...
            n_sample, 100*training_sample_fraction);
    
    fprintf('Training Random Forest: %d trees...\n', num_trees);
    tic;
    model = TreeBagger(num_trees, X_train, y_train, ...
                      'Method', 'regression', ...
                      'OOBPrediction', 'on', ...
                      'MinLeafSize', min_leaf_size, ...
                      'NumPredictorsToSample', mtry, ...
                      'Options', statset('Display', 'iter', 'UseParallel', true));
    train_time = toc;
    fprintf('Training completed in %.2f seconds\n', train_time);
    
    y_train_pred = predictRandomForest(model, X_train);
    
    calibration_params = [];
    if use_calibration && ~isempty(cal_data)
        X_cal = [cal_data.BiasedP, cal_data.Mean, cal_data.Skewness, ...
                 cal_data.Variance, cal_data.Kurtosis];
        y_cal = cal_data.UnbiasedP;
        y_cal_pred = predictRandomForest(model, X_cal);
        calibration_params = learnCalibration(y_cal, y_cal_pred);
    end
    
    X_test = [test_data.BiasedP, test_data.Mean, test_data.Skewness, ...
              test_data.Variance, test_data.Kurtosis];
    y_test = test_data.UnbiasedP;
    y_test_pred_raw = predictRandomForest(model, X_test);
    
    if use_calibration && ~isempty(calibration_params)
        y_test_pred_cal = applyCalibration(y_test_pred_raw, calibration_params);
    else
        y_test_pred_cal = y_test_pred_raw;
    end
    
    % Calculate metrics (simplified - copy full version from your script)
    test_critical_idx_raw = (y_test <= 0.05) & (y_test_pred_raw <= 0.05);
    test_rmse_raw = sqrt(mean((y_test_pred_raw(test_critical_idx_raw) - y_test(test_critical_idx_raw)).^2));
    test_r2_raw = 1 - sum((y_test(test_critical_idx_raw) - y_test_pred_raw(test_critical_idx_raw)).^2) / ...
                  sum((y_test(test_critical_idx_raw) - mean(y_test(test_critical_idx_raw))).^2);
    test_fpr_raw = sum(y_test_pred_raw <= 0.05) / length(y_test_pred_raw);
    
    % Store results
    results = struct();
    results.train_data = train_data;
    results.test_data = test_data;
    results.test_pred_raw = y_test_pred_raw;
    results.test_pred_cal = y_test_pred_cal;
    results.test_critical_idx_raw = test_critical_idx_raw;
    results.test_rmse_raw = test_rmse_raw;
    results.test_r2_raw = test_r2_raw;
    results.test_fpr_raw = test_fpr_raw;
    results.use_calibration = use_calibration;
    
    fprintf('Test RMSE: %.6f, R²: %.4f, FPR: %.4f\n', test_rmse_raw, test_r2_raw, test_fpr_raw);
end

function predictions = predictRandomForest(model, X)
    pred_result = predict(model, X);
    if iscell(pred_result)
        predictions = cell2mat(pred_result);
    else
        predictions = pred_result;
    end
end

function calibration_params = learnCalibration(true_values, predicted_values, cal_range_min, cal_range_max)
    % Simplified version - copy full implementation from your script
    if nargin < 3
        cal_range_min = 0.04;
    end
    if nargin < 4
        cal_range_max = 0.10;
    end
    
    range_idx = (true_values >= cal_range_min & true_values <= cal_range_max) & ...
                (predicted_values >= cal_range_min & predicted_values <= cal_range_max);
    
    if sum(range_idx) < 10
        calibration_params = struct('slope', 1, 'intercept', 0, 'n_samples', 0);
        return;
    end
    
    y_true_range = true_values(range_idx);
    y_pred_range = predicted_values(range_idx);
    
    p = polyfit(y_pred_range, y_true_range, 1);
    
    calibration_params = struct();
    calibration_params.slope = p(1);
    calibration_params.intercept = p(2);
    calibration_params.n_samples = sum(range_idx);
    calibration_params.range_min = cal_range_min;
    calibration_params.range_max = cal_range_max;
end

function calibrated_p = applyCalibration(raw_predictions, calibration_params)
    if calibration_params.n_samples == 0
        calibrated_p = raw_predictions;
        return;
    end
    
    calibrated_p = raw_predictions;
    cal_range_idx = raw_predictions <= calibration_params.range_max;
    
    if sum(cal_range_idx) > 0
        calibrated_subset = calibration_params.slope * raw_predictions(cal_range_idx) + calibration_params.intercept;
        calibrated_subset = max(0, min(1, calibrated_subset));
        calibrated_p(cal_range_idx) = calibrated_subset;
    end
end

function visualizeResults(data, results, model, model_title)
    % Visualize train/test results with focus on critical p-value region
    
    if nargin < 4
        model_title = 'Random Forest';
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
    sample_size = round(0.05 * n_test);
    sample_idx = randperm(n_test, sample_size);
    
    fprintf('Using %d samples (5%%) for visualization, full %d samples for statistics\n', sample_size, n_test);
    
    % Create sampled data for plotting
    test_data_sample = results.test_data(sample_idx, :);
    test_predictions_sample = test_predictions(sample_idx);
    
    % Calculate segment counts using FULL TEST SET on FULL RANGE (0-1)
    y_test = results.test_data.UnbiasedP;
    y_test_pred = test_predictions;
    alpha = 0.05;
    
    seg_A_full = sum(y_test_pred < alpha & y_test >= alpha);  % False positives
    seg_B_full = sum(y_test_pred >= alpha & y_test < alpha);  % False negatives
    seg_C_full = sum(y_test_pred < alpha & y_test < alpha);   % True positives
    seg_D_full = sum(y_test_pred >= alpha & y_test >= alpha); % True negatives
    
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
    
    % Reference lines at 0.05 threshold
    xline(0.05, 'r-', 'LineWidth', 2, 'DisplayName', 'α = 0.05 Threshold');
    yline(0.05, 'r-', 'LineWidth', 2, 'HandleVisibility', 'off');
    
    % Add segment count annotations
    text(0.025, 0.025, sprintf('C: %d', seg_C_full), ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'white', ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.6]);
    
    text(0.025, 0.065, sprintf('A: %d', seg_A_full), ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'white', ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.6]);
    
    text(0.065, 0.025, sprintf('B: %d', seg_B_full), ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'white', ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.6]);
    
    text(0.065, 0.065, sprintf('D: %d', seg_D_full), ...
        'FontSize', 11, 'FontWeight', 'bold', 'Color', 'white', ...
        'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.6]);

    xlabel('Predicted Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Desired Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('QQ Plot: Critical Region (0-0.05)', 'FontSize', 14, 'FontWeight', 'bold');
    xlim([0, 0.075]);
    ylim([0, 0.075]);
    grid on;
    axis square;
    legend('Location', 'southeast', 'FontSize', 10);
    
    % Full range QQ plot with FLIPPED AXES and SMALLER POINTS
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
    
    % Residual plot
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
    xline(0, 'r--', 'LineWidth', 2);
    xlabel('Residuals (Pred - True)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Desired Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('Residual Plot', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    
    % Distribution characteristics
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
    
    % Performance metrics by distribution: RMSE and FPR
    subplot(2, 3, 6);
    dist_rmse = zeros(nDistributions, 1);
    dist_n_critical = zeros(nDistributions, 1);
    dist_fpr = zeros(nDistributions, 1);
    dist_n_total = zeros(nDistributions, 1);
    
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(results.test_data.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            y_true_dist = results.test_data.UnbiasedP(dist_idx);
            y_pred_dist = test_predictions(dist_idx);
            
            % FPR calculation
            total_samples = length(y_pred_dist);
            if total_samples > 0
                dist_fpr(i) = sum(y_pred_dist <= 0.05) / total_samples;
                dist_n_total(i) = total_samples;
            else
                dist_fpr(i) = NaN;
                dist_n_total(i) = 0;
            end
                                    
            % RMSE on critical region
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
    
    % Create grouped bar chart
    bar_data = [dist_rmse, dist_fpr];
    b = bar(bar_data, 'grouped');
    
    if length(b) == 1
        b.FaceColor = 'flat';
        b.CData = [0.3 0.6 0.9; 0.9 0.3 0.3];
    else
        b(1).FaceColor = [0.3 0.6 0.9];
        b(2).FaceColor = [0.9 0.3 0.3];
    end
    
    set(gca, 'XTickLabel', unique_distributions);
    xtickangle(45);
    ylabel('Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('Performance Metrics by Distribution', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'RMSE', 'FPR'}, 'Location', 'northwest');
    grid on;
    
    % Add sample size annotations
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
    
    % Main title
    sgtitle(sprintf('%s Performance: Test RMSE=%.6f, Test R²=%.4f, Overall FPR=%.4f', ...
            model_title, test_rmse, test_r2, test_fpr), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % Save
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = sprintf('rf_results_plot_%s.png', timestamp);
    saveas(gcf, filename);
    fprintf('Saved: %s\n', filename);
    close(gcf);
end