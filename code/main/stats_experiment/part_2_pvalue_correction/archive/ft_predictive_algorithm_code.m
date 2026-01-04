% Trains a polynomial correction model on FieldTrip-generated p-values
% Based on distribution moments (mean, variance, skewness, kurtosis)
% Uses 80/20 train/test split with stratification by distribution
% NOW SUPPORTS PARQUET FORMAT for faster I/O

clear; close all; clc;

%% Global Configuration
global TIME_WINDOW;
TIME_WINDOW = [0.056, 0.100];  % Time window for analysis

%% Setup paths (adjust these to match your system)
main_path = 'C:\Users\CDoga\Documents\Research\PhD\participant_';
fieldtrip_path = 'C:\Users\CDoga\Documents\Research\fieldtrip-20240214';

% Add FieldTrip to path if not already added
if ~exist('ft_defaults', 'file')
    addpath(fieldtrip_path);
    ft_defaults;
end

%% Parameters
nExperimentsPerDist = 25;    % P-values per distribution configuration (can increase for more robust training)
nParticipants = 20;         % Participants per group
ftPermutations = 500;       % FieldTrip permutations

%% Set random seed for reproducibility
rng(42);

%% Load existing data or generate new
load_existing = false;  % Set to true to load existing data
data_filename = 'fieldtrip_pvalue_data_enhanced_2025-07-20_17-56-57.parquet';  % Specify parquet filename here
%data_filename = 'full_combined_fieldtrip_pvalue_data_theoretical_moments_2025-07-14_21-46-23.parquet';
if load_existing
    fprintf('Loading existing data from: %s\n', data_filename);
    data = loadParquetData(data_filename);
    fprintf('Data loaded successfully! Found %d samples\n', height(data));
    n = min(100000, height(data));
    idx = sort(randperm(height(data), n));
    data = data(idx, :);
    fprintf('Sampled down to %d samples\n', height(data));
else
    %% Load Reference Data Structure
    fprintf('Loading reference participant data...\n');
    [reference_data, n_channels, n_timepoints] = load_reference_data(main_path);
    
    cd("C:\Users\CDoga\Documents\Research\EEG-Neuroscience-PhD\code\main\stats_experiment\part_2_pvalue_correction")
    
    %% Generate Training Data
    fprintf('Generating training data using FieldTrip...\n');
    
    % Start parallel pool if not already running
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('Starting parallel pool...\n');
        parpool();
    end
    
    data = generateFieldTripData(nExperimentsPerDist, nParticipants, reference_data, ftPermutations);
    fprintf('Generated %d samples\n', height(data));
end

%% Investigate moments vs biased p-values relationship
fprintf('\n=== INVESTIGATING MOMENTS VS BIASED P-VALUES ===\n');
%investigateMomentsVsPValues(data);

%% Perform Train/Test Split and Model Training
fprintf('\nPerforming stratified 80/20 train/test split...\n');
fprintf('Training full features model...\n');

[results, model] = trainTestSplit(data);

%% Visualize Results
fprintf('\nVisualizing results...\n');
visualizeResults(data, results, model, 'Full Features Model');

%% NEW: 3D Feature Visualization
fprintf('\nCreating 3D feature visualizations...\n');

% Example usage with different feature combinations
plot3DFeatures(results, {'BiasedP', 'Skewness', 'BiasedP×Mean'});
plot3DFeatures(results, {'BiasedP', 'Skewness', 'Mean'});
plot3DFeatures(results, {'BiasedP', 'Skewness', 'Kurtosis'});

%% NEW 3D VISUALIZATION FUNCTION
function plot3DFeatures(results, feature_names)
    % 3D visualization of selected polynomial features for train/test data
    % 
    % Inputs:
    %   results - struct containing training results with X_train_poly and X_test_poly
    %   feature_names - cell array of 3 feature names to visualize
    %
    % Example usage:
    %   plot3DFeatures(results, {'BiasedP', 'Mean', 'Skewness'});
    %   plot3DFeatures(results, {'BiasedP', 'BiasedP²', 'Mean²'});
    
    % Validate input
    if length(feature_names) ~= 3
        error('Exactly 3 feature names must be provided');
    end
    
    % Define complete feature mapping (must match createPolynomialFeatures order)
    % Linear terms (cols 1-5), BiasedP polynomial (cols 6-7), then interactions (cols 8-15)
    all_feature_names = {
        'BiasedP', 'Mean', 'Skewness', 'Variance', 'Kurtosis', ...
        'BiasedP²', 'BiasedP³', ...
        'Mean²', ...
        'BiasedP×Mean', ...
        'Skewness²', ...
        'BiasedP×Skewness', ...
        'Variance²', ...
        'BiasedP×Variance', ...
        'Kurtosis²', ...
        'BiasedP×Kurtosis'
    };
    
    % Find column indices for requested features
    feature_indices = zeros(1, 3);
    for i = 1:3
        idx = find(strcmp(all_feature_names, feature_names{i}));
        if isempty(idx)
            error('Feature "%s" not found. Available features: %s', ...
                  feature_names{i}, strjoin(all_feature_names, ', '));
        end
        feature_indices(i) = idx;
    end
    
    % Extract the 3 selected features from polynomial feature matrices
    X_train_selected = results.X_train_poly(:, feature_indices);
    X_test_selected = results.X_test_poly(:, feature_indices);
    
    % Combine training and testing data
    X_combined = [X_train_selected; X_test_selected];
    
    % Create combined data type labels (train vs test)
    train_labels = repmat({'Train'}, size(X_train_selected, 1), 1);
    test_labels = repmat({'Test'}, size(X_test_selected, 1), 1);
    data_type_labels = [train_labels; test_labels];
    
    % Create figure
    figure('Position', [100, 100, 800, 700]);
    hold on;
    
    % Plot training data
    scatter3(X_train_selected(:, 1), X_train_selected(:, 2), X_train_selected(:, 3), ...
             60, [0.2 0.6 0.9], 'filled', 'MarkerFaceAlpha', 0.6, ...
             'DisplayName', sprintf('Train (n=%d)', size(X_train_selected, 1)));
    
    % Plot test data with different color and marker
    scatter3(X_test_selected(:, 1), X_test_selected(:, 2), X_test_selected(:, 3), ...
             60, [0.9 0.4 0.2], '^', 'filled', 'MarkerFaceAlpha', 0.6, ...
             'DisplayName', sprintf('Test (n=%d)', size(X_test_selected, 1)));
    
    xlabel(feature_names{1}, 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(feature_names{2}, 'FontSize', 12, 'FontWeight', 'bold');
    zlabel(feature_names{3}, 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('Combined 3D Feature Space: %s vs %s vs %s', ...
          feature_names{1}, feature_names{2}, feature_names{3}), ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 11);
    grid on;
    view(45, 30);  % Default 3D view angle
    
    % Print feature statistics
    fprintf('\n=== 3D FEATURE STATISTICS ===\n');
    fprintf('Features: %s, %s, %s\n', feature_names{1}, feature_names{2}, feature_names{3});
    fprintf('Combined data range:\n');
    for i = 1:3
        fprintf('  %s: [%.4f, %.4f]\n', feature_names{i}, ...
                min(X_combined(:, i)), max(X_combined(:, i)));
    end
    fprintf('Training samples: %d, Test samples: %d, Total: %d\n', ...
            size(X_train_selected, 1), size(X_test_selected, 1), size(X_combined, 1));
    fprintf('==============================\n');
end

%% NEW PARQUET I/O FUNCTIONS
function data = loadParquetData(filename)
    % Load data from Parquet file
    % Handle both .parquet and .mat files for backwards compatibility
    
    [~, ~, ext] = fileparts(filename);
    
    switch lower(ext)
        case '.parquet'
            fprintf('Loading Parquet file: %s\n', filename);
            full_data = parquetread(filename);
            
            % Define the features we want to keep
            core_features = {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Distribution'};
            
            % Check which features exist and keep only those
            available_features = {};
            for i = 1:length(core_features)
                if ismember(core_features{i}, full_data.Properties.VariableNames)
                    available_features{end+1} = core_features{i};
                else
                    fprintf('Warning: Feature %s not found in data\n', core_features{i});
                end
            end
            
            % Keep only the core features
            data = full_data(:, available_features);
            
            fprintf('Loaded %d samples with features: %s\n', height(data), strjoin(available_features, ', '));
            
        case '.mat'
            fprintf('Loading MAT file: %s\n', filename);
            loaded = load(filename);
            full_data = loaded.data;
            
            % Same feature selection for MAT files
            core_features = {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Distribution'};
            
            available_features = {};
            for i = 1:length(core_features)
                if ismember(core_features{i}, full_data.Properties.VariableNames)
                    available_features{end+1} = core_features{i};
                else
                    fprintf('Warning: Feature %s not found in data\n', core_features{i});
                end
            end
            
            data = full_data(:, available_features);
            
        otherwise
            error('Unsupported file format: %s. Use .parquet or .mat', ext);
    end
end

function saveParquetData(data, base_filename, nExperimentsPerDist, nParticipants, ftPermutations)
    % Save data in Parquet format only
    % Add metadata as table columns to preserve all information in single file
    
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    
    % Add metadata columns to the data table
    data.nExperimentsPerDist = repmat(nExperimentsPerDist, height(data), 1);
    data.nParticipants = repmat(nParticipants, height(data), 1);
    data.ftPermutations = repmat(ftPermutations, height(data), 1);
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

%% Core Functions

function [reference_data, n_channels, n_timepoints] = load_reference_data(main_path)
    % Load reference data structure from participant 1
    data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
    
    participant_path = strcat(main_path, '1');
    if ~exist(participant_path, 'dir')
        error('Could not find participant directory: %s', participant_path);
    end
    
    cd(participant_path);
    cd('time_domain');
    
    if ~exist(data_file, 'file')
        error('Could not find data file: %s', data_file);
    end
    
    load(data_file);
    reference_data = {};
    reference_data{1} = struct();
    reference_data{1}.label = data.label;
    reference_data{1}.time = data.time{1};
    reference_data{1}.elec = data.elec;
    reference_data{1}.dimord = 'chan_time';
    reference_data{1}.avg = data.med - (data.thin + data.thick)/2;  % PGI calculation
    
    n_channels = size(reference_data{1}.avg, 1);
    n_timepoints = size(reference_data{1}.avg, 2);
    
    fprintf('Reference data loaded: %d channels, %d timepoints\n', n_channels, n_timepoints);
end

function data = generateFieldTripData(nExperimentsPerDist, nParticipants, reference_data, ftPermutations)
    % Generate diverse distributions and calculate FieldTrip p-values
    
    % Distribution configurations with different skewness levels
    distributions = struct();
    %distributions.NORMAL = struct('name', 'NORMAL', 'params', struct());
    %distributions.LAPLACE = struct('name', 'LAPLACE', 'params', struct());
    %distributions.STUDENT_T_3 = struct('name', 'STUDENT_T_3', 'params', struct());
    %distributions.UNIFORM = struct('name', 'UNIFORM', 'params', struct());
    distributions.GAMMA_0_5_1 = struct('name', 'GAMMA_0.5_1', 'params', struct());
    %distributions.EXPONENTIAL = struct('name', 'EXPONENTIAL', 'params', struct());
    %distributions.LOGNORMAL = struct('name', 'LOGNORMAL', 'params', struct());
    dist_names = fieldnames(distributions);
    nDistributions = length(dist_names);
    
    % Get dimensions from reference data
    n_channels = size(reference_data{1}.avg, 1);
    time_vector = reference_data{1}.time;
    time_indices = get_time_indices(time_vector);
    n_timepoints = length(time_indices);
    
    % Preallocate storage for ALL data from ALL distributions
    allBiasedP = [];
    allMoments = [];
    allDistributionLabels = [];
    
    fprintf('  Processing %d distributions with %d experiments each...\n', nDistributions, nExperimentsPerDist);
    fprintf('  Each experiment will generate %d samples (%d channels × %d timepoints)\n', ...
            n_channels * n_timepoints, n_channels, n_timepoints);
    
    % STEP 1: Collect ALL biased p-values from ALL distributions
    for i = 1:nDistributions
        dist_name = dist_names{i};
        dist_info = distributions.(dist_name);
        
        fprintf('  Processing %s distribution...\n', dist_name);
        
        % Storage for this distribution
        biasedP_dist = [];
        moments_dist = [];
        
        % Generate biased p-values from this distribution
        for j = 1:nExperimentsPerDist
            if mod(j, 10) == 0
                fprintf('    Experiment %d/%d for %s\n', j, nExperimentsPerDist, dist_name);
            end
            
            % Generate data from specified distribution
            sampled_data = generate_sampled_data(nParticipants, reference_data, dist_info);
            zero_data = generate_zero_data(nParticipants, reference_data);
            
            % Get FieldTrip p-values matrix (channels × timepoints)
            ft_pvals_matrix = run_fieldtrip_montecarlo(zero_data, sampled_data, ftPermutations);
            
            % PARALLELIZED MOMENTS CALCULATION
            % Extract all participant values first
            participant_values_3d = zeros(n_channels, n_timepoints, nParticipants);
            for p = 1:nParticipants
                participant_values_3d(:, :, p) = sampled_data{p}.avg(:, time_indices);
            end
            
            % Calculate total number of locations
            total_locations = n_channels * n_timepoints;
            
            % Preallocate arrays for this experiment
            exp_pvals = zeros(total_locations, 1);
            exp_moments = zeros(total_locations, 4);
            
            % Use parfor to parallelize across channel-time locations
            parfor loc_idx = 1:total_locations
                % Convert linear index to channel and timepoint indices
                [ch, tp] = ind2sub([n_channels, n_timepoints], loc_idx);
                
                % Extract participant values for this location
                participant_vals = squeeze(participant_values_3d(ch, tp, :));
                
                % Calculate moments for these participant values
                moments_vec = calculateMoments(participant_vals);
                
                % Store results
                exp_pvals(loc_idx) = ft_pvals_matrix(ch, tp);
                exp_moments(loc_idx, :) = moments_vec;
            end
            
            % Accumulate results from this experiment
            biasedP_dist = [biasedP_dist; exp_pvals];
            moments_dist = [moments_dist; exp_moments];
        end
        
        % Create distribution labels for this data
        dist_labels = repmat({dist_name}, length(biasedP_dist), 1);
        
        % Accumulate ALL results across distributions
        allBiasedP = [allBiasedP; biasedP_dist];
        allMoments = [allMoments; moments_dist];
        allDistributionLabels = [allDistributionLabels; dist_labels];
        
        fprintf('  Completed %s distribution: %d samples\n', dist_name, length(biasedP_dist));
    end
    
    % STEP 2: Generate ALL unbiased p-values at once
    fprintf('\n  Generating unbiased p-values for ALL %d samples...\n', length(allBiasedP));
    
    % Generate true uniform p-values (same number as total biased p-values)
    allUnbiasedP = rand(length(allBiasedP), 1);
    
    % STEP 3: Sort and pair by rank (critical for proper calibration)
    fprintf('  Sorting and pairing p-values by rank...\n');
    
    [biasedP_sorted, biasIdx] = sort(allBiasedP);
    [unbiasedP_sorted, ~] = sort(allUnbiasedP);
    moments_sorted = allMoments(biasIdx, :);
    dist_labels_sorted = allDistributionLabels(biasIdx);
    
    % Create final table
    data = table(biasedP_sorted, unbiasedP_sorted, moments_sorted(:,1), moments_sorted(:,2), ...
                 moments_sorted(:,3), moments_sorted(:,4), dist_labels_sorted, ...
        'VariableNames', {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', ...
                          'Skewness', 'Kurtosis', 'Distribution'});
    
    fprintf('  Total samples generated: %d\n', height(data));
    fprintf('  Unbiased p-value range: [%.6f, %.6f]\n', min(unbiasedP_sorted), max(unbiasedP_sorted));
    
    % Save the data using PARQUET format
    fprintf('\n  Saving data...\n');
    saveParquetData(data, 'fieldtrip_pvalue_data', nExperimentsPerDist, nParticipants, ftPermutations);
end

function [results, model] = trainTestSplit(data)
    % Perform stratified 80/20 train/test split and train model
    
    unique_distributions = unique(data.Distribution);
    nDistributions = length(unique_distributions);
    
    % Initialize indices
    train_idx = false(height(data), 1);
    test_idx = false(height(data), 1);
    
    % Stratified split by distribution
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_indices = find(strcmp(data.Distribution, dist_name));
        n_samples = length(dist_indices);
        
        % Randomly permute indices for this distribution
        perm_indices = dist_indices(randperm(n_samples));
        
        % 80% for training, 20% for testing
        n_train = round(0.8 * n_samples);
        
        train_idx(perm_indices(1:n_train)) = true;
        test_idx(perm_indices(n_train+1:end)) = true;
        
        fprintf('Distribution %s: %d train, %d test\n', dist_name, n_train, n_samples - n_train);
    end
    
    % Split data
    train_data = data(train_idx, :);
    test_data = data(test_idx, :);
    
    fprintf('\nTotal: %d train samples, %d test samples\n', height(train_data), height(test_data));
    
    % Train model using all features
    fprintf('Training model using all features...\n');
    X_train = [train_data.BiasedP, train_data.Mean, train_data.Skewness, train_data.Variance, train_data.Kurtosis];
    X_test = [test_data.BiasedP, test_data.Mean, test_data.Skewness, test_data.Variance, test_data.Kurtosis];
    X_train_poly = createPolynomialFeatures(X_train);
    X_test_poly = createPolynomialFeatures(X_test);
    
    y_train = train_data.UnbiasedP;
    y_test = test_data.UnbiasedP;
    
    % Train model
    model = fitlm(X_train_poly, y_train);
    
    % Evaluate on training set - CRITICAL REGION ONLY
    y_train_pred = predict(model, X_train_poly);
    train_critical_idx = (y_train <= 0.05) & (y_train_pred <= 0.05);
    
    if sum(train_critical_idx) > 0
        train_rmse = sqrt(mean((y_train_pred(train_critical_idx) - y_train(train_critical_idx)).^2));
        train_r2 = 1 - sum((y_train(train_critical_idx) - y_train_pred(train_critical_idx)).^2) / sum((y_train(train_critical_idx) - mean(y_train(train_critical_idx))).^2);
        train_mae = mean(abs(y_train_pred(train_critical_idx) - y_train(train_critical_idx)));
    else
        train_rmse = NaN; train_r2 = NaN; train_mae = NaN;
    end
    
    % Evaluate on test set - CRITICAL REGION ONLY
    y_test_pred = predict(model, X_test_poly);
    test_critical_idx = (y_test <= 0.05) & (y_test_pred <= 0.05);
    
    if sum(test_critical_idx) > 0
        test_rmse = sqrt(mean((y_test_pred(test_critical_idx) - y_test(test_critical_idx)).^2));
        test_r2 = 1 - sum((y_test(test_critical_idx) - y_test_pred(test_critical_idx)).^2) / sum((y_test(test_critical_idx) - mean(y_test(test_critical_idx))).^2);
        test_mae = mean(abs(y_test_pred(test_critical_idx) - y_test(test_critical_idx)));
    else
        test_rmse = NaN; test_r2 = NaN; test_mae = NaN;
    end
    
    % Store results
    results = struct();
    results.train_data = train_data;
    results.test_data = test_data;
    results.train_pred = y_train_pred;
    results.test_pred = y_test_pred;
    results.train_critical_idx = train_critical_idx;
    results.test_critical_idx = test_critical_idx;
    results.train_rmse = train_rmse;
    results.train_r2 = train_r2;
    results.train_mae = train_mae;
    results.test_rmse = test_rmse;
    results.test_r2 = test_r2;
    results.test_mae = test_mae;
    results.X_train_poly = X_train_poly;  % Store polynomial features for plotting
    results.X_test_poly = X_test_poly;
    
    % Print results
    fprintf('\n========== MODEL PERFORMANCE (CRITICAL REGION ≤ 0.05) ==========\n');
    fprintf('Model type: Full feature set (BiasedP + Moments)\n');
    fprintf('Training Set (n=%d critical samples):\n', sum(train_critical_idx));
    fprintf('  RMSE: %.6f, R²: %.4f, MAE: %.6f\n', train_rmse, train_r2, train_mae);
    fprintf('Test Set (n=%d critical samples):\n', sum(test_critical_idx));
    fprintf('  RMSE: %.6f, R²: %.4f, MAE: %.6f\n', test_rmse, test_r2, test_mae);
    fprintf('Model R² (full data): %.4f\n', model.Rsquared.Ordinary);
    
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
    
    unique_distributions = unique(data.Distribution);
    nDistributions = length(unique_distributions);
    colors = lines(nDistributions);
    
    % Create figure with larger size
    figure('Position', [50, 50, 2000, 1400]);
    
    % Main QQ plot focused on critical region (0-0.05)
    subplot(2, 3, [1, 2]);
    
    % Filter test data to critical region
    critical_idx = (results.test_data.UnbiasedP <= 0.05) & (results.test_pred <= 0.05);
    y_true_critical = results.test_data.UnbiasedP(critical_idx);
    y_pred_critical = results.test_pred(critical_idx);
    test_dist_critical = results.test_data.Distribution(critical_idx);
    
    % Plot by distribution
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(test_dist_critical, dist_name);
        
        if sum(dist_idx) > 0
            scatter(y_true_critical(dist_idx), y_pred_critical(dist_idx), 60, colors(i,:), 'filled', ...
                   'MarkerFaceAlpha', 0.7, 'DisplayName', dist_name);
            hold on;
        end
    end
    
    % Perfect prediction line
    plot([0, 0.05], [0, 0.05], 'r--', 'LineWidth', 3, 'DisplayName', 'Perfect Prediction');
    
    xlabel('True P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Predicted P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('QQ Plot: Critical Region (0-0.05)', 'FontSize', 14, 'FontWeight', 'bold');
    xlim([0, 0.05]);
    ylim([0, 0.05]);
    grid on;
    axis square;
    legend('Location', 'southeast', 'FontSize', 10);
    
    % Full range QQ plot
    subplot(2, 3, 3);
    
    % Plot by distribution for full range
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(results.test_data.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            scatter(results.test_data.UnbiasedP(dist_idx), results.test_pred(dist_idx), 40, colors(i,:), 'filled', ...
                   'MarkerFaceAlpha', 0.7, 'DisplayName', dist_name);
            hold on;
        end
    end
    
    % Perfect prediction line
    plot([0, 1], [0, 1], 'r--', 'LineWidth', 3, 'DisplayName', 'Perfect Prediction');
    
    xlabel('True P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Predicted P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('QQ Plot: Full Range', 'FontSize', 14, 'FontWeight', 'bold');
    xlim([0, 1]);
    ylim([0, 1]);
    grid on;
    axis square;
    legend('Location', 'southeast', 'FontSize', 10);
    
    % Residual plot
    subplot(2, 3, 4);
    residuals = results.test_pred - results.test_data.UnbiasedP;
    
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(results.test_data.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            scatter(results.test_data.UnbiasedP(dist_idx), residuals(dist_idx), 40, colors(i,:), 'filled', ...
                   'MarkerFaceAlpha', 0.7, 'DisplayName', dist_name);
            hold on;
        end
    end
    
    yline(0, 'r--', 'LineWidth', 2);
    xlabel('True P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Residuals (Pred - True)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Residual Plot', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    
    % Distribution characteristics
    subplot(2, 3, 5);
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(results.test_data.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            test_data_dist = results.test_data(dist_idx, :);
            
            % Filter out invalid values
            valid_idx = ~isnan(test_data_dist.Skewness) & ~isnan(test_data_dist.Kurtosis) & ...
                       ~isinf(test_data_dist.Skewness) & ~isinf(test_data_dist.Kurtosis) & ...
                       abs(test_data_dist.Skewness) < 10 & abs(test_data_dist.Kurtosis) < 50;
            
            if sum(valid_idx) > 0
                scatter(test_data_dist.Skewness(valid_idx), test_data_dist.Kurtosis(valid_idx), ...
                       40, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.7, 'DisplayName', dist_name);
                hold on;
            end
        end
    end
    
    xlabel('Skewness', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Kurtosis', 'FontSize', 12, 'FontWeight', 'bold');
    title('Distribution Characteristics', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    legend('Location', 'best', 'FontSize', 10);
    
    % Performance metrics - CRITICAL REGION ONLY
    subplot(2, 3, 6);
    
    % Calculate metrics by distribution for critical region only
    dist_rmse = zeros(nDistributions, 1);
    dist_r2 = zeros(nDistributions, 1);
    dist_n_critical = zeros(nDistributions, 1);
    
    for i = 1:nDistributions
        dist_name = unique_distributions{i};
        dist_idx = strcmp(results.test_data.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            y_true_dist = results.test_data.UnbiasedP(dist_idx);
            y_pred_dist = results.test_pred(dist_idx);
            
            % Focus on critical region for this distribution
            critical_idx = (y_true_dist <= 0.05) & (y_pred_dist <= 0.05);
            dist_n_critical(i) = sum(critical_idx);
            
            if sum(critical_idx) > 0
                y_true_critical = y_true_dist(critical_idx);
                y_pred_critical = y_pred_dist(critical_idx);
                
                dist_rmse(i) = sqrt(mean((y_pred_critical - y_true_critical).^2));
                dist_r2(i) = 1 - sum((y_true_critical - y_pred_critical).^2) / sum((y_true_critical - mean(y_true_critical)).^2);
            else
                dist_rmse(i) = NaN;
                dist_r2(i) = NaN;
            end
        end
    end
    
    % Create bar plot with sample sizes
    bar(dist_rmse, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'black', 'LineWidth', 1);
    set(gca, 'XTickLabel', unique_distributions);
    xtickangle(45);
    ylabel('RMSE', 'FontSize', 12, 'FontWeight', 'bold');
    title('RMSE by Distribution (Critical Region ≤ 0.05)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    % Add text annotations showing sample sizes
    for i = 1:nDistributions
        if ~isnan(dist_rmse(i))
            text(i, dist_rmse(i) + 0.0001, sprintf('n=%d', dist_n_critical(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 9);
        end
    end
    
    % Main title - CRITICAL REGION METRICS
    sgtitle(sprintf('%s Performance (Critical Region ≤ 0.05): Test RMSE=%.6f, Test R²=%.4f', ...
            model_name, results.test_rmse, results.test_r2), ...
            'FontSize', 16, 'FontWeight', 'bold');
end

function data = generate_zero_data(n_participants, reference_data)
    % Generate zero data for control group
    data = cell(n_participants, 1);
    
    n_channels = size(reference_data{1}.avg, 1);
    n_timepoints = size(reference_data{1}.avg, 2);
    
    for p = 1:n_participants
        participant_data = reference_data{1};
        participant_data.avg = zeros(n_channels, n_timepoints);
        data{p} = participant_data;
    end
end

function data = generate_sampled_data(n_participants, reference_data, dist_info)
    % Generate sampled data for experimental group
    data = cell(n_participants, 1);
    
    n_channels = size(reference_data{1}.avg, 1);
    n_timepoints = size(reference_data{1}.avg, 2);
    
    for p = 1:n_participants
        participant_data = reference_data{1};
        participant_data.avg = generate_distribution_data(n_channels, n_timepoints, dist_info);
        data{p} = participant_data;
    end
end

function data_matrix = generate_distribution_data(n_channels, n_size, dist_info)
    % Generate n_channels x n_size matrix from specified distribution
    % dist_info.name contains the distribution type
    
    % Total number of samples needed
    n_total = n_channels * n_size;
    
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
            
        case 'STUDENT_T_3'
            df = 3;
            sample = trnd(df, n_total, 1);
            
        case 'UNIFORM'
            min_val = -0.5;
            max_val = 0.5;
            sample = min_val + (max_val - min_val) * rand(n_total, 1);
            
        case 'GAMMA_0.5_1'
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
    
    % Reshape the vector into n_channels x n_size matrix
    data_matrix = reshape(sample, n_channels, n_size);
end

function moments = calculateMoments(x)
    % Calculate four moments from data
    m = mean(x);
    v = var(x);
    s = skewness(x);
    k = kurtosis(x) - 3;  % Excess kurtosis
    moments = [m, v, s, k];
end

function time_indices = get_time_indices(time_vector)
    % Get time indices for the specified time window
    global TIME_WINDOW;
    
    time_indices = find(time_vector >= TIME_WINDOW(1) & time_vector <= TIME_WINDOW(2));
    
    if isempty(time_indices)
        error('No time points found in the specified time window [%.3f, %.3f]', TIME_WINDOW(1), TIME_WINDOW(2));
    end
end

function all_pvalues = run_fieldtrip_montecarlo(zero_data, sampled_data, n_permutations)
    % Run FieldTrip Monte Carlo method
    global TIME_WINDOW;
    
    n_participants = length(zero_data);
    
    % Create design matrix
    design_matrix = [1:n_participants 1:n_participants; ones(1,n_participants) 2*ones(1,n_participants)];
    
    % Setup neighbours for cluster correction
    cfg = [];
    cfg.feedback = 'no';
    cfg.method = 'distance';
    cfg.elec = zero_data{1}.elec;
    neighbours = ft_prepare_neighbours(cfg);
    
    % Configure FieldTrip statistics
    cfg = [];
    cfg.latency = TIME_WINDOW;
    cfg.channel = 'eeg';
    cfg.statistic = 'ft_statfun_depsamplesT';
    cfg.method = 'montecarlo';
    cfg.correctm = 'cluster';
    cfg.neighbours = neighbours;
    cfg.clusteralpha = 0.025;
    cfg.numrandomization = n_permutations;
    cfg.tail = 0;
    cfg.design = design_matrix;
    cfg.computeprob = 'no';
    cfg.alpha = 0.05;
    cfg.correcttail = 'alpha';
    cfg.clusterthreshold = 'nonparametric_individual';
    cfg.uvar = 1;
    cfg.ivar = 2;
    
    % Run FieldTrip statistics
    stat = ft_timelockstatistics(cfg, zero_data{:}, sampled_data{:});
    
    % Return p-values matrix (channels × timepoints)
    all_pvalues = stat.prob;
end

function investigateMomentsVsPValues(data)
    % Investigate the relationship between moments and both biased and unbiased p-values
    
    % Calculate correlations for both biased and unbiased p-values
    corr_mean_biased = corr(data.Mean, data.BiasedP);
    corr_var_biased = corr(data.Variance, data.BiasedP);
    corr_skew_biased = corr(data.Skewness, data.BiasedP);
    corr_kurt_biased = corr(data.Kurtosis, data.BiasedP);
    
    corr_mean_unbiased = corr(data.Mean, data.UnbiasedP);
    corr_var_unbiased = corr(data.Variance, data.UnbiasedP);
    corr_skew_unbiased = corr(data.Skewness, data.UnbiasedP);
    corr_kurt_unbiased = corr(data.Kurtosis, data.UnbiasedP);
    
    fprintf('Correlations with BiasedP:\n');
    fprintf('  Mean: %.4f, Variance: %.4f, Skewness: %.4f, Kurtosis: %.4f\n', ...
        corr_mean_biased, corr_var_biased, corr_skew_biased, corr_kurt_biased);
    
    fprintf('Correlations with UnbiasedP:\n');
    fprintf('  Mean: %.4f, Variance: %.4f, Skewness: %.4f, Kurtosis: %.4f\n', ...
        corr_mean_unbiased, corr_var_unbiased, corr_skew_unbiased, corr_kurt_unbiased);
    
    % Create simple scatter plots - 2 rows (biased vs unbiased), 4 columns (moments)
    figure('Position', [50, 50, 1600, 900]);
    
    % Row 1: BIASED P-VALUES
    subplot(2,4,1);
    scatter(data.Mean, data.BiasedP, 20, 'filled');
    xlabel('Mean'); ylabel('Biased P-Value');
    title(sprintf('Mean vs BiasedP (r=%.3f)', corr_mean_biased));
    grid on;
    
    subplot(2,4,2);
    scatter(data.Variance, data.BiasedP, 20, 'filled');
    xlabel('Variance'); ylabel('Biased P-Value');
    title(sprintf('Variance vs BiasedP (r=%.3f)', corr_var_biased));
    grid on;
    
    subplot(2,4,3);
    scatter(data.Skewness, data.BiasedP, 20, 'filled');
    xlabel('Skewness'); ylabel('Biased P-Value');
    title(sprintf('Skewness vs BiasedP (r=%.3f)', corr_skew_biased));
    grid on;
    
    subplot(2,4,4);
    scatter(data.Kurtosis, data.BiasedP, 20, 'filled');
    xlabel('Kurtosis'); ylabel('Biased P-Value');
    title(sprintf('Kurtosis vs BiasedP (r=%.3f)', corr_kurt_biased));
    grid on;
    
    % Row 2: UNBIASED P-VALUES
    subplot(2,4,5);
    scatter(data.Mean, data.UnbiasedP, 20, 'filled');
    xlabel('Mean'); ylabel('Unbiased P-Value');
    title(sprintf('Mean vs UnbiasedP (r=%.3f)', corr_mean_unbiased));
    grid on;
    
    subplot(2,4,6);
    scatter(data.Variance, data.UnbiasedP, 20, 'filled');
    xlabel('Variance'); ylabel('Unbiased P-Value');
    title(sprintf('Variance vs UnbiasedP (r=%.3f)', corr_var_unbiased));
    grid on;
    
    subplot(2,4,7);
    scatter(data.Skewness, data.UnbiasedP, 20, 'filled');
    xlabel('Skewness'); ylabel('Unbiased P-Value');
    title(sprintf('Skewness vs UnbiasedP (r=%.3f)', corr_skew_unbiased));
    grid on;
    
    subplot(2,4,8);
    scatter(data.Kurtosis, data.UnbiasedP, 20, 'filled');
    xlabel('Kurtosis'); ylabel('Unbiased P-Value');
    title(sprintf('Kurtosis vs UnbiasedP (r=%.3f)', corr_kurt_unbiased));
    grid on;
    
    sgtitle('Moments vs P-Values Investigation: Biased (Row 1) vs Unbiased (Row 2)', 'FontSize', 16, 'FontWeight', 'bold');
    
    fprintf('=== END INVESTIGATION ===\n\n');
end