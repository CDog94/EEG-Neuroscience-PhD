function [models, predictions, training_info] = model_builder(data, cfg)
    % BUILD MODEL(S) AND MAKE PREDICTIONS
    % Supports both single-model and range-specific training
    % Returns models, predictions on test set, and training info
    
    if cfg.range_specific.enabled
        fprintf('\n=== RANGE-SPECIFIC TRAINING ===\n');
        [models, predictions, training_info] = trainRangeSpecificModels(data, cfg);
    else
        fprintf('\n=== SINGLE MODEL TRAINING ===\n');
        [models, predictions, training_info] = trainSingleModel(data, cfg);
    end
end

%% ============================================================================
%% LOCAL FUNCTIONS (Helper functions used by build_and_predict)
%% ============================================================================

function [model, predictions, info] = trainSingleModel(data, cfg)
    % Sample training data if requested
    n_train_samples = height(data.train);
    n_sample = round(cfg.training.training_sample_fraction * n_train_samples);
    sample_idx = randperm(n_train_samples, n_sample);
    train_data_sample = data.train(sample_idx, :);
    
    fprintf('Using %d samples (%.1f%% of training data)\n', n_sample, ...
            cfg.training.training_sample_fraction * 100);
    
    % Extract features and target
    if strcmp(cfg.model.type, 'polynomial')
        X_train = extractPolynomialFeatures(train_data_sample);
    else
        X_train = extractFeatures(train_data_sample);
    end
    y_train = train_data_sample.UnbiasedP;
    
    % Train model based on type
    fprintf('Training %s model...\n', cfg.model.type);
    tic;
    model = trainModel(X_train, y_train, cfg.model.type, cfg.model.params);
    train_time = toc;
    fprintf('Training completed in %.2f seconds\n', train_time);
    
    % Make predictions on test set
    if strcmp(cfg.model.type, 'polynomial')
        X_test = extractPolynomialFeatures(data.test);
    else
        X_test = extractFeatures(data.test);
    end
    predictions_raw = predictModel(model, X_test, cfg.model.type);
    
    % Apply calibration if enabled
    if cfg.training.use_calibration && ~isempty(data.cal)
        fprintf('Learning calibration parameters...\n');
        if strcmp(cfg.model.type, 'polynomial')
            X_cal = extractPolynomialFeatures(data.cal);
        else
            X_cal = extractFeatures(data.cal);
        end
        y_cal = data.cal.UnbiasedP;
        predictions_cal = predictModel(model, X_cal, cfg.model.type);
        
        cal_params = learnCalibration(y_cal, predictions_cal, cfg.calibration);
        predictions = applyCalibration(predictions_raw, cal_params);
        
        info.calibration_params = cal_params;
        info.predictions_raw = predictions_raw;
    else
        predictions = predictions_raw;
        info.calibration_params = [];
        info.predictions_raw = predictions_raw;
    end
    
    % Store training info
    info.train_time = train_time;
    info.n_train_samples = n_sample;
    info.model_type = cfg.model.type;
    info.model_assignment = ones(length(predictions), 1);  % All use same model
end

%% ============================================================================
%% RANGE-SPECIFIC TRAINING
%% ============================================================================

function [models, predictions, info] = trainRangeSpecificModels(data, cfg)
    n_ranges = length(cfg.range_specific.training_ranges);
    fprintf('Training %d separate models for different BiasedP ranges\n', n_ranges);
    
    models = cell(n_ranges, 1);
    train_samples_per_range = zeros(n_ranges, 1);
    
    % Train model for each range
    for r = 1:n_ranges
        range = cfg.range_specific.training_ranges{r};
        fprintf('\n=== Range %d: [%.4f, %.4f] ===\n', r, range(1), range(2));
        
        % Filter training data by range
        range_idx = (data.train.BiasedP >= range(1)) & (data.train.BiasedP <= range(2));
        train_data_range = data.train(range_idx, :);
        train_samples_per_range(r) = height(train_data_range);
        
        fprintf('Training samples in range: %d (%.1f%% of full training set)\n', ...
                height(train_data_range), 100*height(train_data_range)/height(data.train));
        
        if height(train_data_range) < 100
            warning('Very few samples (%d) in range [%.4f, %.4f]', ...
                    height(train_data_range), range(1), range(2));
        end
        
        % Sample if needed
        n_sample = max(100, round(cfg.training.training_sample_fraction * height(train_data_range)));
        n_sample = min(n_sample, height(train_data_range));
        sample_idx = randperm(height(train_data_range), n_sample);
        train_data_sample = train_data_range(sample_idx, :);
        
        % Train model
        if strcmp(cfg.model.type, 'polynomial')
            X_train = extractPolynomialFeatures(train_data_sample);
        else
            X_train = extractFeatures(train_data_sample);
        end
        y_train = train_data_sample.UnbiasedP;
        
        fprintf('Training %s with %d samples...\n', cfg.model.type, n_sample);
        tic;
        models{r} = trainModel(X_train, y_train, cfg.model.type, cfg.model.params);
        fprintf('Range %d training completed in %.2f seconds\n', r, toc);
    end
    
    % Make predictions on test set
    fprintf('\n=== Making Predictions on Test Set ===\n');
    n_test = height(data.test);
    predictions_raw = zeros(n_test, 1);
    model_assignment = zeros(n_test, 1);
    
    if strcmp(cfg.model.type, 'polynomial')
        X_test = extractPolynomialFeatures(data.test);
    else
        X_test = extractFeatures(data.test);
    end
    
    % Assign and predict for each range
    for r = 1:n_ranges
        range = cfg.range_specific.training_ranges{r};
        in_range_idx = (data.test.BiasedP >= range(1)) & (data.test.BiasedP <= range(2));
        n_in_range = sum(in_range_idx);
        
        if n_in_range > 0
            X_test_range = X_test(in_range_idx, :);
            predictions_raw(in_range_idx) = predictModel(models{r}, X_test_range, cfg.model.type);
            model_assignment(in_range_idx) = r;
            fprintf('  Range %d [%.4f, %.4f]: %d predictions\n', r, range(1), range(2), n_in_range);
        end
    end
    
    % Handle out-of-range samples
    out_of_range_idx = (model_assignment == 0);
    n_out_of_range = sum(out_of_range_idx);
    
    if n_out_of_range > 0
        fprintf('Processing %d out-of-range samples using %s strategy...\n', ...
                n_out_of_range, cfg.range_specific.out_of_range_strategy);
        
        switch cfg.range_specific.out_of_range_strategy
            case 'nearest'
                % Find nearest range for each sample
                out_biased_p = data.test.BiasedP(out_of_range_idx);
                nearest_model = findNearestRange(out_biased_p, cfg.range_specific.training_ranges);
                
                % Predict using nearest models
                for r = 1:n_ranges
                    use_this_model = (nearest_model == r);
                    if any(use_this_model)
                        oor_indices = find(out_of_range_idx);
                        global_indices = oor_indices(use_this_model);
                        X_nearest = X_test(global_indices, :);
                        predictions_raw(global_indices) = predictModel(models{r}, X_nearest, cfg.model.type);
                        model_assignment(global_indices) = r;
                    end
                end
                
            case 'passthrough'
                predictions_raw(out_of_range_idx) = data.test.BiasedP(out_of_range_idx);
                model_assignment(out_of_range_idx) = 0;
        end
    end
    
    % Apply calibration if enabled
    if cfg.training.use_calibration && ~isempty(data.cal)
        fprintf('\n=== Learning Calibration Per Range ===\n');
        cal_params = cell(n_ranges, 1);
        
        for r = 1:n_ranges
            range = cfg.range_specific.training_ranges{r};
            range_cal_idx = (data.cal.BiasedP >= range(1)) & (data.cal.BiasedP <= range(2));
            
            if sum(range_cal_idx) >= 10
                if strcmp(cfg.model.type, 'polynomial')
                    X_cal = extractPolynomialFeatures(data.cal(range_cal_idx, :));
                else
                    X_cal = extractFeatures(data.cal(range_cal_idx, :));
                end
                y_cal = data.cal.UnbiasedP(range_cal_idx);
                y_cal_pred = predictModel(models{r}, X_cal, cfg.model.type);
                cal_params{r} = learnCalibration(y_cal, y_cal_pred, cfg.calibration);
            else
                cal_params{r} = struct('slope', 1, 'intercept', 0, 'n_samples', 0);
            end
        end
        
        % Apply calibration
        predictions = predictions_raw;
        for r = 1:n_ranges
            range_test_idx = (model_assignment == r);
            if sum(range_test_idx) > 0 && cal_params{r}.n_samples > 0
                predictions(range_test_idx) = applyCalibration(predictions_raw(range_test_idx), cal_params{r});
            end
        end
        
        info.calibration_params = cal_params;
        info.predictions_raw = predictions_raw;
    else
        predictions = predictions_raw;
        info.calibration_params = [];
        info.predictions_raw = predictions_raw;
    end
    
    % Store info
    info.model_assignment = model_assignment;
    info.training_ranges = cfg.range_specific.training_ranges;
    info.train_samples_per_range = train_samples_per_range;
    info.model_type = cfg.model.type;
end

%% ============================================================================
%% MODEL TRAINING (DISPATCHER)
%% ============================================================================

function model = trainModel(X, y, model_type, params)
    switch model_type
        case 'random_forest'
            model = trainRandomForest(X, y, params);
            
        case 'svr'
            model = trainSVR(X, y, params);
            
        case 'polynomial'
            model = trainPolynomial(X, y, params);
            
        case 'gbm'
            error('GBM not yet implemented');
            
        otherwise
            error('Unknown model type: %s', model_type);
    end
end

%% ============================================================================
%% MODEL-SPECIFIC TRAINING FUNCTIONS
%% ============================================================================

function model = trainRandomForest(X, y, params)
    model = TreeBagger(params.num_trees, X, y, ...
                      'Method', 'regression', ...
                      'OOBPrediction', 'on', ...
                      'MinLeafSize', params.min_leaf_size, ...
                      'NumPredictorsToSample', params.mtry, ...
                      'Options', statset('Display', 'off', 'UseParallel', true));
    
    fprintf('OOB Error: %.6f\n', oobError(model, 'Mode', 'Ensemble'));
    
    % Print feature importances
    printRandomForestImportances(model);
end

function model = trainSVR(X, y, params)
    % Set default parameters if not provided
    if ~isfield(params, 'kernel_function')
        params.kernel_function = 'gaussian';
    end
    if ~isfield(params, 'box_constraint')
        params.box_constraint = 1;
    end
    if ~isfield(params, 'epsilon')
        params.epsilon = 0.01;
    end
    if ~isfield(params, 'kernel_scale')
        params.kernel_scale = 'auto';
    end
    if ~isfield(params, 'standardize')
        params.standardize = true;
    end
    
    fprintf('Training SVR: kernel=%s, C=%.2f, epsilon=%.3f\n', ...
            params.kernel_function, params.box_constraint, params.epsilon);
    
    % Train SVR model
    model = fitrsvm(X, y, ...
                   'KernelFunction', params.kernel_function, ...
                   'BoxConstraint', params.box_constraint, ...
                   'Epsilon', params.epsilon, ...
                   'KernelScale', params.kernel_scale, ...
                   'Standardize', params.standardize);
    
    fprintf('SVR training complete. Number of support vectors: %d\n', size(model.SupportVectors, 1));
end

function model = trainPolynomial(X, y, params)
    % X is already transformed with polynomial features
    % params currently unused (for future regularization)
    
    fprintf('Training polynomial regression with %d features...\n', size(X, 2));
    model = fitlm(X, y);
    
    fprintf('Polynomial R² (training): %.4f\n', model.Rsquared.Ordinary);
    
    % Print the model equation
    printPolynomialEquation(model);
end

%% ============================================================================
%% MODEL PREDICTION (DISPATCHER)
%% ============================================================================

function predictions = predictModel(model, X, model_type)
    switch model_type
        case 'random_forest'
            predictions = predictRandomForest(model, X);
            
        case 'svr'
            predictions = predictSVR(model, X);
            
        case 'polynomial'
            predictions = predictPolynomial(model, X);
            
        otherwise
            error('Unknown model type: %s', model_type);
    end
end

function predictions = predictRandomForest(model, X)
    pred_result = predict(model, X);
    if iscell(pred_result)
        predictions = cell2mat(pred_result);
    else
        predictions = pred_result;
    end
end

function predictions = predictSVR(model, X)
    predictions = predict(model, X);
end

function predictions = predictPolynomial(model, X)
    % X is already transformed with polynomial features
    predictions = predict(model, X);
    
    % Clip to valid p-value range [0, 1]
    predictions = max(0, min(1, predictions));
end

%% ============================================================================
%% CALIBRATION
%% ============================================================================

function cal_params = learnCalibration(y_true, y_pred, cal_cfg)
    % Learn linear calibration in specified range
    range_idx = (y_true >= cal_cfg.range_min & y_true <= cal_cfg.range_max) & ...
                (y_pred >= cal_cfg.range_min & y_pred <= cal_cfg.range_max);
    
    if sum(range_idx) < 10
        cal_params = struct('slope', 1, 'intercept', 0, 'n_samples', 0, ...
                           'range_min', cal_cfg.range_min, 'range_max', cal_cfg.range_max);
        fprintf('  Insufficient calibration samples (%d)\n', sum(range_idx));
        return;
    end
    
    y_true_range = y_true(range_idx);
    y_pred_range = y_pred(range_idx);
    
    % Fit linear model
    p = polyfit(y_pred_range, y_true_range, 1);
    
    cal_params = struct();
    cal_params.slope = p(1);
    cal_params.intercept = p(2);
    cal_params.n_samples = sum(range_idx);
    cal_params.range_min = cal_cfg.range_min;
    cal_params.range_max = cal_cfg.range_max;
    
    fprintf('  Calibration learned: y = %.4f*x + %.4f (n=%d)\n', ...
            cal_params.slope, cal_params.intercept, cal_params.n_samples);
end

function calibrated = applyCalibration(predictions, cal_params)
    if cal_params.n_samples == 0
        calibrated = predictions;
        return;
    end
    
    calibrated = predictions;
    cal_range_idx = predictions <= cal_params.range_max;
    
    if sum(cal_range_idx) > 0
        calibrated_subset = cal_params.slope * predictions(cal_range_idx) + cal_params.intercept;
        calibrated_subset = max(0, min(1, calibrated_subset));
        calibrated(cal_range_idx) = calibrated_subset;
    end
end

%% ============================================================================
%% UTILITY FUNCTIONS
%% ============================================================================

function X = extractFeatures(data)
    % Extract feature matrix from data table (for RF/SVR)
    X = [data.BiasedP, data.Mean, data.Skewness, data.Variance, data.Kurtosis];
end

function X_poly = extractPolynomialFeatures(data)
    % Extract base features
    X = [data.BiasedP, data.Mean, data.Skewness, data.Variance, data.Kurtosis];
    
    % Apply polynomial transformation
    X_poly = createPolynomialFeatures(X);
end

function X_poly = createPolynomialFeatures(X)
    % Create polynomial features from base features
    % X columns: [BiasedP, Mean, Skewness, Variance, Kurtosis]
    % Returns 16 features total
    
    X_poly = [X, ...                              % Linear terms (5)
             X(:,1).^2, X(:,1).^3, ...           % BiasedP² and BiasedP³ (2)
             X(:,2).^2, ...                      % Mean² (1)
             X(:,1).*X(:,2), ...                 % BiasedP × Mean (1)
             X(:,3).^2, ...                      % Skewness² (1)
             X(:,1).*X(:,3), ...                 % BiasedP × Skewness (1)
             X(:,4).^2, ...                      % Variance² (1)
             X(:,1).*X(:,4), ...                 % BiasedP × Variance (1)
             X(:,5).^2, ...                      % Kurtosis² (1)
             X(:,1).*X(:,5)];                    % BiasedP × Kurtosis (1)
    % Total: 5 + 11 = 16 features
end

function nearest_model = findNearestRange(biased_p_values, training_ranges)
    % Find nearest range for each p-value
    n = length(biased_p_values);
    n_ranges = length(training_ranges);
    
    min_distances = inf(n, 1);
    nearest_model = ones(n, 1);
    
    for r = 1:n_ranges
        range = training_ranges{r};
        
        % Calculate distance to this range
        distances = zeros(n, 1);
        below = biased_p_values < range(1);
        above = biased_p_values > range(2);
        
        distances(below) = range(1) - biased_p_values(below);
        distances(above) = biased_p_values(above) - range(2);
        
        % Update nearest range
        closer = distances < min_distances;
        min_distances(closer) = distances(closer);
        nearest_model(closer) = r;
    end
end

function printPolynomialEquation(model)
    % Print the polynomial regression equation in readable format
    fprintf('\n--- POLYNOMIAL EQUATION ---\n');
    
    % Feature names matching createPolynomialFeatures output
    feature_names = {'BiasedP', 'Mean', 'Skewness', 'Variance', 'Kurtosis', ...
                     'BiasedP²', 'BiasedP³', 'Mean²', 'BiasedP×Mean', ...
                     'Skewness²', 'BiasedP×Skewness', 'Variance²', 'BiasedP×Variance', ...
                     'Kurtosis²', 'BiasedP×Kurtosis'};
    
    coeffs = model.Coefficients.Estimate;
    intercept = coeffs(1);
    
    % Start with intercept
    fprintf('UnbiasedP = %.6f', intercept);
    
    % Add each term
    for i = 2:length(coeffs)
        coeff = coeffs(i);
        feat_idx = i - 1;  % Offset by 1 because first coeff is intercept
        
        if feat_idx <= length(feature_names)
            feat_name = feature_names{feat_idx};
        else
            feat_name = sprintf('x%d', feat_idx);
        end
        
        if coeff >= 0
            fprintf(' + %.6f*%s', coeff, feat_name);
        else
            fprintf(' - %.6f*%s', abs(coeff), feat_name);
        end
        
        % Add line break every 3 terms for readability
        if mod(i-1, 3) == 0 && i < length(coeffs)
            fprintf('\n            ');
        end
    end
    fprintf('\n---------------------------\n\n');
end

function printRandomForestImportances(model)
    % Print feature importance rankings for random forest
    fprintf('\n--- RANDOM FOREST FEATURE IMPORTANCES ---\n');
    
    % Feature names for base features used by RF
    feature_names = {'BiasedP', 'Mean', 'Skewness', 'Variance', 'Kurtosis'};
    
    % Get out-of-bag permuted predictor importance estimates
    % Use property access for compatibility with older MATLAB versions
    try
        importances = model.OOBPermutedPredictorDeltaError;
    catch
        fprintf('Feature importance not available for this TreeBagger model.\n');
        fprintf('------------------------------------------\n\n');
        return;
    end
    
    % Sort by importance (descending)
    [sorted_imp, sort_idx] = sort(importances, 'descend');
    
    fprintf('Rank  Feature          Importance\n');
    fprintf('----  ---------------  ----------\n');
    for i = 1:length(sorted_imp)
        feat_idx = sort_idx(i);
        if feat_idx <= length(feature_names)
            feat_name = feature_names{feat_idx};
        else
            feat_name = sprintf('Feature_%d', feat_idx);
        end
        
        fprintf('%2d    %-15s  %.6f\n', i, feat_name, sorted_imp(i));
    end
    fprintf('------------------------------------------\n\n');
end