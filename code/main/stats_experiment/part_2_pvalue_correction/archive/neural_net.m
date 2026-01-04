%% Simple Neural Network for P-value Correction
% GPU-accelerated dense neural network for correcting permutation test p-values
% Research-focused implementation with exact visualization from baseline

clear; close all; clc;

%% ============================================================================
%% CONFIGURATION
%% ============================================================================

% Data parameters
DATA_SUBSET_SIZE = 5000000000000000;  % Use 5M samples for faster iteration
DATA_FILENAME = 'two_sample.parquet';  % Change to 'sign_swap.parquet' if needed

% Neural network hyperparameters
BATCH_SIZE = 65536;  % Larger batch for RTX 4080
LEARNING_RATE = 0.0001;
MAX_EPOCHS = 3;
VALIDATION_FREQUENCY = 1;  % Validate every N epochs

% Set random seed
rng(42);

%% ============================================================================
%% GPU CHECK
%% ============================================================================

fprintf('========================================\n');
fprintf('GPU INITIALIZATION\n');
fprintf('========================================\n');

if ~canUseGPU()
    error('GPU not available. This script requires GPU acceleration.');
end

g = gpuDevice();
fprintf('GPU: %s (%.1f GB available)\n', g.Name, g.AvailableMemory/1e9);
reset(g);  % Clear GPU memory

%% ============================================================================
%% LOAD AND PREPARE DATA
%% ============================================================================

fprintf('\n========================================\n');
fprintf('DATA LOADING\n');
fprintf('========================================\n');

% Load parquet data
fprintf('Loading %s...\n', DATA_FILENAME);
tic;
data = parquetread(DATA_FILENAME);
load_time = toc;
fprintf('Loaded %d samples in %.1f seconds\n', height(data), load_time);

% Take subset for faster iteration
if height(data) > DATA_SUBSET_SIZE
    fprintf('Sampling %d from %d total samples...\n', DATA_SUBSET_SIZE, height(data));
    subset_idx = randperm(height(data), DATA_SUBSET_SIZE);
    data = data(subset_idx, :);
end

% Extract features and targets
X = [data.BiasedP, data.Mean, data.Variance, data.Skewness, data.Kurtosis];
y = data.UnbiasedP;

% Normalize features (important for neural networks)
fprintf('Normalizing features...\n');
mu = mean(X, 1);
sigma = std(X, 0, 1);
sigma(sigma == 0) = 1;  % Prevent division by zero
X_normalized = (X - mu) ./ sigma;

% Report critical region statistics for reference
critical_idx = data.BiasedP <= 0.05;
fprintf('Critical region samples (p<=0.05): %d (%.1f%%)\n', ...
        sum(critical_idx), 100*mean(critical_idx));

%% ============================================================================
%% TRAIN/TEST SPLIT (STRATIFIED BY DISTRIBUTION)
%% ============================================================================

fprintf('\n========================================\n');
fprintf('STRATIFIED TRAIN/TEST SPLIT\n');
fprintf('========================================\n');

% Stratified split by distribution - each distribution appears in both train and test
unique_distributions = unique(data.Distribution);
nDistributions = length(unique_distributions);

train_idx = false(height(data), 1);
test_idx = false(height(data), 1);

for i = 1:nDistributions
    dist_name = unique_distributions{i};
    dist_indices = find(strcmp(data.Distribution, dist_name));
    n_samples = length(dist_indices);
    
    % Randomly permute indices
    perm_indices = dist_indices(randperm(n_samples));
    
    % 80/20 split
    n_train = round(0.8 * n_samples);
    n_test = n_samples - n_train;
    
    % Assign indices
    train_idx(perm_indices(1:n_train)) = true;
    test_idx(perm_indices(n_train+1:end)) = true;
    
    fprintf('  %s: %d train, %d test\n', dist_name, n_train, n_test);
end

% Create train/test sets
X_train = X_normalized(train_idx, :);
y_train = y(train_idx);
train_data = data(train_idx, :);

X_test = X_normalized(test_idx, :);
y_test = y(test_idx);
test_data = data(test_idx, :);

fprintf('Total: %d train, %d test\n', sum(train_idx), sum(test_idx));

%% ============================================================================
%% BUILD NEURAL NETWORK
%% ============================================================================

fprintf('\n========================================\n');
fprintf('BUILDING NEURAL NETWORK\n');
fprintf('========================================\n');

layers = [
    featureInputLayer(5, 'Name', 'input', 'Normalization', 'none')  % We already normalized

    fullyConnectedLayer(512, 'Name', 'fc1')
    reluLayer('Name', 'relu1')

    fullyConnectedLayer(256, 'Name', 'fc2')
    reluLayer('Name', 'relu2')

    fullyConnectedLayer(128, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.3, 'Name', 'dropout1')

    fullyConnectedLayer(64, 'Name', 'fc4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.3, 'Name', 'dropout2')

    fullyConnectedLayer(32, 'Name', 'fc5')
    reluLayer('Name', 'relu5')

    fullyConnectedLayer(1, 'Name', 'fc_output')
    sigmoidLayer('Name', 'output')
    regressionLayer('Name', 'regression')
];

% layers = [
%     featureInputLayer(5, 'Name', 'input', 'Normalization', 'none')
% 
%     % Single hidden layer with moderate size
%     fullyConnectedLayer(64, 'Name', 'fc1')
%     reluLayer('Name', 'relu1')
%     dropoutLayer(0.2, 'Name', 'dropout1')  % Reduced dropout
% 
%     % Smaller second layer
%     fullyConnectedLayer(32, 'Name', 'fc2')
%     reluLayer('Name', 'relu2')
%     dropoutLayer(0.1, 'Name', 'dropout2')  % Even less dropout
% 
%     % Output layer - NO SIGMOID! Let regression layer handle it
%     fullyConnectedLayer(1, 'Name', 'fc_output')
%     sigmoidLayer('Name', 'output')
%     regressionLayer('Name', 'regression')
% ];

% Display network architecture
fprintf('Network architecture:\n');
fprintf('  Input: 5 features (normalized)\n');
fprintf('  Hidden: 256 → 128 → 64 → 32 neurons\n');
fprintf('  Activation: ReLU + Dropout(0.3)\n');
fprintf('  Output: 1 neuron (sigmoid activation)\n');
fprintf('  Total parameters: ~44K\n');

%% ============================================================================
%% TRAINING OPTIONS
%% ============================================================================

fprintf('\n========================================\n');
fprintf('CONFIGURING TRAINING\n');
fprintf('========================================\n');

options = trainingOptions('adam', ...
    'MaxEpochs', MAX_EPOCHS, ...
    'MiniBatchSize', BATCH_SIZE, ...
    'InitialLearnRate', LEARNING_RATE, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', {X_test, y_test}, ...
    'ValidationFrequency', 500, ...
    'ValidationPatience', 10, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 50);

fprintf('Training configuration:\n');
fprintf('  Batch size: %d\n', BATCH_SIZE);
fprintf('  Initial learning rate: %.4f\n', LEARNING_RATE);
fprintf('  Max epochs: %d\n', MAX_EPOCHS);
fprintf('  Optimizer: Adam\n');
fprintf('  Validation every: 500 iterations\n');
fprintf('  Early stopping patience: 10 validation checks\n');
fprintf('  Equal treatment: All p-value regions weighted equally\n');

%% ============================================================================
%% TRAIN NETWORK
%% ============================================================================

fprintf('\n========================================\n');
fprintf('TRAINING NEURAL NETWORK\n');
fprintf('========================================\n');

% Train with original data (no sample weighting or duplication)
train_start = tic;
net = trainNetwork(X_train, y_train, layers, options);

train_time = toc(train_start);
fprintf('\nTraining completed in %.1f minutes\n', train_time/60);

%% ============================================================================
%% EVALUATE MODEL
%% ============================================================================

fprintf('\n========================================\n');
fprintf('MODEL EVALUATION\n');
fprintf('========================================\n');

% Predictions on test set
y_pred_test = predict(net, X_test, 'ExecutionEnvironment', 'gpu');

% Ensure predictions are in [0, 1]
y_pred_test = max(0, min(1, y_pred_test));

% Create results structure for visualization function
results = struct();
results.test_data = test_data;
results.test_pred = y_pred_test;
results.test_pred_cal = y_pred_test;  % No calibration for NN
results.use_calibration = false;

% Calculate metrics for critical region
test_critical_idx = (test_data.UnbiasedP <= 0.05) & (y_pred_test <= 0.05);
results.test_critical_idx = test_critical_idx;
results.test_critical_idx_cal = test_critical_idx;

if sum(test_critical_idx) > 0
    y_true_critical = test_data.UnbiasedP(test_critical_idx);
    y_pred_critical = y_pred_test(test_critical_idx);
    
    results.test_rmse = sqrt(mean((y_pred_critical - y_true_critical).^2));
    results.test_r2 = 1 - sum((y_true_critical - y_pred_critical).^2) / ...
                      sum((y_true_critical - mean(y_true_critical)).^2);
    results.test_rmse_cal = results.test_rmse;
    results.test_r2_cal = results.test_r2;
else
    results.test_rmse = NaN;
    results.test_r2 = NaN;
    results.test_rmse_cal = NaN;
    results.test_r2_cal = NaN;
end

% Calculate FPR
false_positives = sum((test_data.UnbiasedP > 0.05) & (y_pred_test <= 0.05));
results.test_fpr = false_positives / height(test_data);

% Print performance summary
fprintf('\n========== NEURAL NETWORK PERFORMANCE ==========\n');
fprintf('Test Set - Critical Region (p ≤ 0.05):\n');
fprintf('  Samples: %d (%.1f%% of test set)\n', sum(test_critical_idx), ...
        100*sum(test_critical_idx)/height(test_data));
fprintf('  RMSE: %.6f\n', results.test_rmse);
fprintf('  R²: %.4f\n', results.test_r2);
fprintf('  FPR: %.4f\n', results.test_fpr);

% Overall R²
r2_overall = 1 - sum((test_data.UnbiasedP - y_pred_test).^2) / ...
             sum((test_data.UnbiasedP - mean(test_data.UnbiasedP)).^2);
fprintf('Test Set - Overall:\n');
fprintf('  R²: %.4f\n', r2_overall);

%% ============================================================================
%% VISUALIZATION (Using exact function from original script)
%% ============================================================================

fprintf('\n========================================\n');
fprintf('GENERATING VISUALIZATIONS\n');
fprintf('========================================\n');

model_name = sprintf('Neural Network (5-256-128-64-32-1) - Equal Treatment');
visualizeResults(data, results, [], model_name);

% Clear GPU memory
reset(g);
fprintf('\n✓ Training and evaluation complete! GPU memory cleared.\n');

%% ============================================================================
%% VISUALIZATION FUNCTION (Exact copy from original script)
%% ============================================================================

function visualizeResults(data, results, model, model_name)
    % Visualize train/test results with focus on critical p-value region
    
    if nargin < 4
        model_name = 'Model';
    end
    
    % Determine which prediction fields to use
    if results.use_calibration && isfield(results, 'test_pred_cal') && isfield(results, 'calibration_params') && ~isempty(results.calibration_params)
        test_predictions = results.test_pred_cal;
        test_critical_idx = results.test_critical_idx_cal;
        test_rmse = results.test_rmse_cal;
        test_r2 = results.test_r2_cal;
        test_fpr = results.test_fpr_cal;
    else
        test_predictions = results.test_pred;
        test_critical_idx = results.test_critical_idx;
        test_rmse = results.test_rmse;
        test_r2 = results.test_r2;
        test_fpr = results.test_fpr;
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
    figure('Position', [50, 50, 2000, 1400]);
    
    % Main QQ plot focused on critical region (0-0.05)
    subplot(2, 3, [1, 2]);
    
    % Filter test data to critical region (using sampled data for plotting)
    critical_idx_sample = (test_data_sample.UnbiasedP <= 0.05) & (test_predictions_sample <= 0.05);
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
    plot([0, 0.05], [0, 0.05], 'r--', 'LineWidth', 3, 'DisplayName', 'Perfect Prediction');
    
    xlabel('Predicted Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Desired Unbiased P-Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('QQ Plot: Critical Region (0-0.05)', 'FontSize', 14, 'FontWeight', 'bold');
    xlim([0, 0.05]);
    ylim([0, 0.05]);
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
            
            % Calculate FPR on FULL RANGE (0-1)
            false_positives = sum((y_true_dist > 0.05) & (y_pred_dist <= 0.05));
            total_samples = length(y_true_dist);
            dist_n_total(i) = total_samples;
            
            if total_samples > 0
                dist_fpr(i) = false_positives / total_samples;
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
    b(1).FaceColor = [0.3 0.6 0.9];
    b(2).FaceColor = [0.9 0.3 0.3];
    
    set(gca, 'XTickLabel', unique_distributions);
    xtickangle(45);
    ylabel('Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('Performance Metrics by Distribution', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'RMSE (Critical Region 0-0.05)', 'FPR (Full Range 0-1)'}, 'Location', 'northwest');
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
    overall_false_positives = sum((results.test_data.UnbiasedP > 0.05) & (test_predictions <= 0.05));
    overall_fpr = overall_false_positives / height(results.test_data);
        
    % Main title with FPR included
    sgtitle(sprintf('%s Performance: Test RMSE=%.6f, Test R²=%.4f, Overall FPR=%.4f', ...
            model_name, test_rmse, test_r2, overall_fpr), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % Save and close
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = sprintf('nn_results_plot_%s.png', timestamp);
    saveas(gcf, filename);
    fprintf('Saved: %s\n', filename);
    close(gcf);
end