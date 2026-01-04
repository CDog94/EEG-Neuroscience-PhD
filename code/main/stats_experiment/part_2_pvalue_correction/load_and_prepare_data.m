function data = load_and_prepare_data(cfg)
    % LOAD AND PREPARE DATA FOR MODEL TRAINING
    % Returns struct with train/test/cal splits and metadata
    % Now also stores raw sample vectors for diagnostic plotting
    parallel.gpu.enableCUDAForwardCompatibility(true);
    
    rng(cfg.random_seed);
    
    %% Load or generate data
    if cfg.data.load_existing
        fprintf('Loading existing data from: %s\n', cfg.data.file);
        [raw_data, raw_samples] = loadData(cfg.data.file);
        fprintf('Data loaded successfully! Found %d samples\n', height(raw_data));
    else
        fprintf('Generating training data using %s permutation tests...\n', upper(cfg.data.test_type));
        tic;
        [raw_data, raw_samples] = generatePermutationDataFast(cfg.data.n_pvalues, cfg.data.n_participants, ...
                                               cfg.data.n_permutations, cfg.data.test_type);
        fprintf('Generated %d samples in %.2f seconds\n', height(raw_data), toc);
    end
    
    %% Apply filtering
    [raw_data, raw_samples] = filterData(raw_data, raw_samples, cfg.filter);
    
    %% Apply augmentation
    if cfg.augment.use_theoretical_moments
        fprintf('\n=== APPLYING THEORETICAL MOMENTS ===\n');
        raw_data = applyTheoreticalMoments(raw_data);
    end
    
    if cfg.augment.oversample.enabled
        fprintf('\n=== APPLYING OVERSAMPLING ===\n');
        [raw_data, raw_samples] = oversampleData(raw_data, raw_samples, cfg.augment.oversample);
    end
    
    %% Perform stratified train/test split
    fprintf('\n=== PERFORMING STRATIFIED SPLIT ===\n');
    [train_data, test_data, cal_data, train_samples, test_samples, cal_samples] = ...
        stratifiedSplit(raw_data, raw_samples, cfg.training);
    
    %% Package results
    data = struct();
    data.train = train_data;
    data.test = test_data;
    data.cal = cal_data;
    data.raw = raw_data;
    data.distributions = unique(raw_data.Distribution);
    
    % Store raw samples
    data.train_samples = train_samples;
    data.test_samples = test_samples;
    data.cal_samples = cal_samples;
    data.raw_samples = raw_samples;
    
    fprintf('\nData preparation complete!\n');
    fprintf('Training samples: %d\n', height(train_data));
    fprintf('Test samples: %d\n', height(test_data));
    if ~isempty(cal_data)
        fprintf('Calibration samples: %d\n', height(cal_data));
    end
end


%% ============================================================================
%% LOCAL FUNCTIONS
%% ============================================================================

function [data, raw_samples] = loadData(filename)
    [filepath, name, ext] = fileparts(filename);
    
    switch lower(ext)
        case '.parquet'
            fprintf('Loading Parquet file: %s\n', filename);
            full_data = parquetread(filename);
            
            % Core features plus any parameter columns (Param_*)
            var_names = full_data.Properties.VariableNames;
            core_features = {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Distribution'};
            param_features = var_names(startsWith(var_names, 'Param_'));
            
            available_features = {};
            for i = 1:length(core_features)
                if ismember(core_features{i}, var_names)
                    available_features{end+1} = core_features{i};
                end
            end
            available_features = [available_features, param_features];
            
            data = full_data(:, available_features);
            fprintf('Loaded %d samples with features: %s\n', height(data), strjoin(available_features, ', '));
            
            % Try to load corresponding raw samples parquet file first
            samples_parquet = fullfile(filepath, [name '_samples.parquet']);
            if exist(samples_parquet, 'file')
                fprintf('Loading raw samples from: %s\n', samples_parquet);
                samples_table = parquetread(samples_parquet);
                raw_samples = table2array(samples_table);
            else
                % Fallback to .mat file for backwards compatibility
                samples_mat = fullfile(filepath, [name '_samples.mat']);
                if exist(samples_mat, 'file')
                    fprintf('Loading raw samples from: %s\n', samples_mat);
                    loaded = load(samples_mat);
                    raw_samples = loaded.raw_samples;
                else
                    warning('No raw samples file found. Diagnostic plots will not be available.');
                    raw_samples = [];
                end
            end
            
        case '.mat'
            fprintf('Loading MAT file: %s\n', filename);
            loaded = load(filename);
            data = loaded.data;
            if isfield(loaded, 'raw_samples')
                raw_samples = loaded.raw_samples;
            else
                raw_samples = [];
            end
            
        otherwise
            error('Unsupported file format: %s. Use .parquet or .mat', ext);
    end
end

%% ============================================================================
%% DATA FILTERING
%% ============================================================================

function [data_filtered, samples_filtered] = filterData(data, raw_samples, filter_cfg)
    fprintf('\n=== APPLYING FILTERS ===\n');
    before_filtering = height(data);
    keep_idx = true(height(data), 1);
    
    % Exclude distributions
    if ~isempty(filter_cfg.distributions_to_exclude)
        fprintf('Excluding selected distributions...\n');
        for i = 1:length(filter_cfg.distributions_to_exclude)
            dist_name = filter_cfg.distributions_to_exclude{i};
            dist_idx = strcmp(data.Distribution, dist_name);
            n_excluded = sum(dist_idx);
            if n_excluded > 0
                fprintf('  Excluding %s: %d samples\n', dist_name, n_excluded);
                keep_idx = keep_idx & ~dist_idx;
            end
        end
    end
    
    % Skewness filter
    if isfinite(filter_cfg.skewness_threshold)
        skewness_filter = abs(data.Skewness) >= filter_cfg.skewness_threshold;
        keep_idx = keep_idx & ~skewness_filter;
        fprintf('Applied skewness filter: |skewness| >= %.1f\n', filter_cfg.skewness_threshold);
    end
    
    data_filtered = data(keep_idx, :);
    if ~isempty(raw_samples)
        samples_filtered = raw_samples(keep_idx, :);
    else
        samples_filtered = [];
    end
    
    fprintf('\nFILTERING RESULTS:\n');
    fprintf('Original: %d samples\n', before_filtering);
    fprintf('Retained: %d samples (%.1f%%)\n', height(data_filtered), 100*height(data_filtered)/before_filtering);
    
    % Show remaining distributions
    remaining_dists = unique(data_filtered.Distribution);
    fprintf('\nRemaining distributions:\n');
    for i = 1:length(remaining_dists)
        dist_name = remaining_dists{i};
        n_samples = sum(strcmp(data_filtered.Distribution, dist_name));
        fprintf('  %s: %d samples (%.1f%%)\n', dist_name, n_samples, 100*n_samples/height(data_filtered));
    end
end

%% ============================================================================
%% DATA AUGMENTATION
%% ============================================================================

function data_modified = applyTheoreticalMoments(data)
    data_modified = data;
    unique_distributions = unique(data.Distribution);
    
    fprintf('Replacing with theoretical moments for %d distributions...\n', length(unique_distributions));
    
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
        
        fprintf('  %-12s: %d samples | [mu=%.4f, var=%.4f, skew=%.4f, kurt=%.4f]\n', ...
                dist_name, n_samples, theoretical_moments(1), theoretical_moments(2), ...
                theoretical_moments(3), theoretical_moments(4));
    end
end

function moments = getTheoreticalMoments(distribution_name)
    switch upper(distribution_name)
        case 'NORMAL'
            moments = [0, 1, 0, 0];
        case 'UNIFORM'
            moments = [0, (0.5 - (-0.5))^2 / 12, 0, -1.2];
        case 'LAPLACE'
            moments = [0, 2, 0, 3];
        case 'STUDENT_T'
            df = 3;
            variance = ternary(df > 2, df / (df - 2), Inf);
            excess_kurtosis = ternary(df > 4, 6 / (df - 4), 6);
            moments = [0, variance, 0, excess_kurtosis];
        case 'GAMMA'
            shape = 0.5; scale = 1;
            moments = [0, shape * scale^2, 2 / sqrt(shape), 6 / shape];
        case 'EXPONENTIAL'
            rate = 1;
            moments = [0, 1 / rate^2, 2, 6];
        case 'LOGNORMAL'
            mu = 0; sigma = 0.5;
            variance = (exp(sigma^2) - 1) * exp(2*mu + sigma^2);
            skewness = (exp(sigma^2) + 2) * sqrt(exp(sigma^2) - 1);
            excess_kurtosis = exp(4*sigma^2) + 2*exp(3*sigma^2) + 3*exp(2*sigma^2) - 6;
            moments = [0, variance, skewness, excess_kurtosis];
        otherwise
            error('Unknown distribution: %s', distribution_name);
    end
end

function [data_oversampled, samples_oversampled] = oversampleData(data, raw_samples, oversample_cfg)
    target_range_idx = (data.BiasedP >= oversample_cfg.range_min & data.BiasedP <= oversample_cfg.range_max) | ...
                       (data.UnbiasedP >= oversample_cfg.range_min & data.UnbiasedP <= oversample_cfg.range_max);
    
    n_target = sum(target_range_idx);
    fprintf('Found %d samples in range [%.3f, %.3f]\n', n_target, oversample_cfg.range_min, oversample_cfg.range_max);
    
    if n_target == 0
        warning('No samples in target range - skipping oversampling');
        data_oversampled = data;
        samples_oversampled = raw_samples;
        return;
    end
    
    target_data = data(target_range_idx, :);
    if ~isempty(raw_samples)
        target_samples = raw_samples(target_range_idx, :);
    end
    
    n_additional = round((oversample_cfg.factor - 1) * n_target);
    fprintf('Generating %d additional samples (%.1fx factor)\n', n_additional, oversample_cfg.factor);
    
    % Bootstrap with noise
    bootstrap_idx = randi(n_target, n_additional, 1);
    additional_data = target_data(bootstrap_idx, :);
    if ~isempty(raw_samples)
        additional_samples = target_samples(bootstrap_idx, :);
    end
    
    % Add small noise to summary stats (excluding parameter columns)
    noise_factor = 0.01;
    continuous_vars = {'BiasedP', 'UnbiasedP', 'Mean', 'Variance', 'Skewness', 'Kurtosis'};
    for i = 1:length(continuous_vars)
        var_name = continuous_vars{i};
        if ismember(var_name, data.Properties.VariableNames)
            var_std = std(target_data.(var_name));
            noise = randn(size(additional_data.(var_name))) * var_std * noise_factor;
            additional_data.(var_name) = additional_data.(var_name) + noise;
            
            if strcmp(var_name, 'BiasedP') || strcmp(var_name, 'UnbiasedP')
                additional_data.(var_name) = max(0, min(1, additional_data.(var_name)));
            elseif strcmp(var_name, 'Variance')
                additional_data.(var_name) = max(0, additional_data.(var_name));
            end
        end
    end
    
    % Add small noise to raw samples too
    if ~isempty(raw_samples)
        sample_std = std(target_samples(:));
        sample_noise = randn(size(additional_samples)) * sample_std * noise_factor;
        additional_samples = additional_samples + sample_noise;
    end
    
    data_oversampled = [data; additional_data];
    perm_idx = randperm(height(data_oversampled));
    data_oversampled = data_oversampled(perm_idx, :);
    
    if ~isempty(raw_samples)
        samples_oversampled = [raw_samples; additional_samples];
        samples_oversampled = samples_oversampled(perm_idx, :);
    else
        samples_oversampled = [];
    end
    
    fprintf('Final dataset: %d samples (%.1fx increase)\n', height(data_oversampled), ...
            height(data_oversampled)/height(data));
end

%% ============================================================================
%% TRAIN/TEST SPLIT
%% ============================================================================

function [train_data, test_data, cal_data, train_samples, test_samples, cal_samples] = ...
         stratifiedSplit(data, raw_samples, training_cfg)
    
    unique_distributions = unique(data.Distribution);
    n_distributions = length(unique_distributions);
    
    train_idx = false(height(data), 1);
    cal_idx = false(height(data), 1);
    test_idx = false(height(data), 1);
    
    train_pct = training_cfg.train_pct;
    cal_pct = training_cfg.cal_pct;
    test_pct = training_cfg.test_pct;
    
    if abs(train_pct + cal_pct + test_pct - 1.0) > 1e-6
        error('Split percentages must sum to 1.0');
    end
    
    fprintf('Split ratios: Train=%.0f%%, Cal=%.0f%%, Test=%.0f%%\n', ...
            train_pct*100, cal_pct*100, test_pct*100);
    
    for i = 1:n_distributions
        dist_name = unique_distributions{i};
        dist_indices = find(strcmp(data.Distribution, dist_name));
        n_samples = length(dist_indices);
        
        perm_indices = dist_indices(randperm(n_samples));
        
        n_train = round(train_pct * n_samples);
        n_cal = round(cal_pct * n_samples);
        
        train_idx(perm_indices(1:n_train)) = true;
        
        if n_cal > 0
            cal_idx(perm_indices(n_train+1:n_train+n_cal)) = true;
            test_idx(perm_indices(n_train+n_cal+1:end)) = true;
        else
            test_idx(perm_indices(n_train+1:end)) = true;
        end
    end
    
    train_data = data(train_idx, :);
    test_data = data(test_idx, :);
    
    if ~isempty(raw_samples)
        train_samples = raw_samples(train_idx, :);
        test_samples = raw_samples(test_idx, :);
    else
        train_samples = [];
        test_samples = [];
    end
    
    if cal_pct > 0
        cal_data = data(cal_idx, :);
        if ~isempty(raw_samples)
            cal_samples = raw_samples(cal_idx, :);
        else
            cal_samples = [];
        end
    else
        cal_data = [];
        cal_samples = [];
    end
end

%% ============================================================================
%% DATA GENERATION WITH RAW SAMPLE AND PARAMETER STORAGE
%% ============================================================================

function [data, raw_samples] = generatePermutationDataFast(nPValues, nParticipants, nPermutations, test_type)

    % GPU forward compatibility
    parallel.gpu.enableCUDAForwardCompatibility(true);

    if gpuDeviceCount > 0
        g = gpuDevice;
        fprintf('Using GPU: %s (%.1f GB free)\n', g.Name, g.AvailableMemory/1e9);
        USE_GPU = true;
    else
        warning('No GPU detected — falling back to CPU.');
        USE_GPU = false;
    end


    %% =================== CONFIG ===================
    BATCH = 25000;   % <<< ------- GPU batch size


    %% =================== DISTRIBUTIONS ===================
    distributions = [
        struct('name', 'EXPONENTIAL', 'params', [])
        struct('name', 'GAMMA',       'params', [])
        struct('name', 'LOGNORMAL',   'params', [])
    ];

    nDistributions = length(distributions);
    nPValuesPerDist = ceil(nPValues / nDistributions);
    actualTotal = nPValuesPerDist * nDistributions;

    fprintf('  Generating %d p-values per distribution (%d distributions)\n', ...
            nPValuesPerDist, nDistributions);
    fprintf('  Actual total: %d p-values\n', actualTotal);


    %% =================== STORAGE ===================
    allBiasedP = zeros(actualTotal, 1);
    allMoments = zeros(actualTotal, 4);
    allDistLabels = zeros(actualTotal, 1);
    allRawSamples = zeros(actualTotal, nParticipants);
    allParams = cell(actualTotal, 1);


    %% =================== PERMUTATION MATRIX ===================
    fprintf('  Pre-generating permutation indices for %s test...\n', test_type);

    switch test_type
        case 'one-sample'
            permIndices = rand(nPermutations, nParticipants) > 0.5;
        case 'paired'
            permIndices = rand(nPermutations, nParticipants) > 0.5;
        otherwise
            error('Invalid test_type: %s', test_type);
    end

    signs = 2 .* permIndices - 1;   % convert to ±1


    %% Upload permutation matrix ONCE
    if USE_GPU
        Sg = gpuArray(signs);
    end


%% =================== MAIN LOOP ===================
idx = 1;

for d = 1:nDistributions

    dist_info = distributions(d);

    fprintf('\n=== Processing distribution %d/%d: %s (%s test) ===\n', ...
        d, nDistributions, dist_info.name, upper(test_type));

    tic;

    % Pre-generate samples + params for this distribution
    dist_samples = zeros(nPValuesPerDist, nParticipants);
    dist_params  = cell(nPValuesPerDist,1);

    for p = 1:nPValuesPerDist
        [sample_data, params_used] = generateDistributionSampleWithParams(nParticipants, dist_info);
        dist_samples(p,:) = sample_data(:)';
        dist_params{p} = params_used;
    end


    %% ---------- GPU STREAMING BATCHES ----------
    dist_pvals   = zeros(nPValuesPerDist,1);
    dist_moments = zeros(nPValuesPerDist,4);

    nBatches = ceil(nPValuesPerDist / BATCH);

    fprintf('Processing %d samples in %d batches of %d...\n', ...
        nPValuesPerDist, nBatches, BATCH);


    for b = 1:BATCH:nPValuesPerDist

        batchIdx = ceil(b / BATCH);
        j = min(b + BATCH - 1, nPValuesPerDist);

        fprintf('  Distribution %s — Batch %d/%d  (samples %d–%d)...', ...
            dist_info.name, batchIdx, nBatches, b, j);
        batch_tic = tic;


        %% ==== LOAD BATCH ====
        X = dist_samples(b:j,:)';   % (participants × batch)
        B = size(X,2);


        %% ==== OBSERVED T ====
        means_obs = mean(X,1);
        std_obs   = std(X,0,1);
        t_obs     = means_obs ./ (std_obs ./ sqrt(nParticipants));


        %% ==== PERMUTATION NULL ====
        if USE_GPU
            Xg = gpuArray(X);

            perm_means = (Sg * Xg) ./ nParticipants;
            denom = sqrt(sum(Xg.^2,1) ./ (nParticipants^2));
            t_perm = perm_means ./ denom;

            counts = sum(abs(t_perm) >= abs(t_obs),1);

            pvals = gather((counts + 1) ./ (nPermutations + 1));

            clear Xg perm_means t_perm denom counts
        else
            perm_means = (signs * X) ./ nParticipants;
            denom = sqrt(sum(X.^2,1) ./ (nParticipants^2));
            t_perm = perm_means ./ denom;
            counts = sum(abs(t_perm) >= abs(t_obs),1);
            pvals = (counts + 1) ./ (nPermutations + 1);
        end


        %% ==== STORE RESULTS ====
        dist_pvals(b:j) = pvals;

        for k = 1:B
            dist_moments(b+k-1,:) = calculateMoments(X(:,k));
        end

        clear X means_obs std_obs t_obs
        wait(gpuDevice);


        fprintf(' done (%.2f sec)\n', toc(batch_tic));
    end



    %% ---------- SAVE FOR THIS DISTRIBUTION ----------
    endIdx = idx + nPValuesPerDist - 1;

    allBiasedP(idx:endIdx) = dist_pvals;
    allMoments(idx:endIdx,:) = dist_moments;
    allRawSamples(idx:endIdx,:) = dist_samples;
    allDistLabels(idx:endIdx) = d;
    allParams(idx:endIdx) = dist_params;

    idx = endIdx + 1;

    fprintf('Completed %s in %.2f sec\n', dist_info.name, toc);
end



    %% =================== UNBIASED P-VALUES ===================
    fprintf('Generating uniform unbiased p-values...\n');

    allUnbiasedP = zeros(actualTotal,1);

    idx = 1;
    for d = 1:nDistributions
        endIdx = idx + nPValuesPerDist - 1;
        uniform_values = ((1:nPValuesPerDist)-0.5)/nPValuesPerDist;
        allUnbiasedP(idx:endIdx) = uniform_values';
        idx = endIdx + 1;
    end


    %% =================== SORT PER DISTRIBUTION ===================
    biasedP_sorted = zeros(actualTotal,1);
    unbiasedP_sorted = zeros(actualTotal,1);
    moments_sorted = zeros(actualTotal,4);
    dist_labels_sorted = zeros(actualTotal,1);
    samples_sorted = zeros(actualTotal,nParticipants);
    params_sorted = cell(actualTotal,1);

    idx = 1;
    for d = 1:nDistributions
        endIdx = idx + nPValuesPerDist - 1;

        [sorted_biased, sort_idx] = sort(allBiasedP(idx:endIdx));

        biasedP_sorted(idx:endIdx) = sorted_biased;
        unbiasedP_sorted(idx:endIdx) = allUnbiasedP(idx:endIdx);
        moments_sorted(idx:endIdx,:) = allMoments(idx-1+sort_idx,:);
        dist_labels_sorted(idx:endIdx) = allDistLabels(idx-1+sort_idx);
        samples_sorted(idx:endIdx,:) = allRawSamples(idx-1+sort_idx,:);
        params_sorted(idx:endIdx) = allParams(idx-1+sort_idx);

        idx = endIdx + 1;
    end


    %% =================== OUTPUT ===================
    dist_names = {distributions.name};
    dist_labels_cell = dist_names(dist_labels_sorted)';

    param_table = extractParameterColumns(params_sorted, dist_labels_cell, dist_names);

    data = table(biasedP_sorted, unbiasedP_sorted, ...
                 moments_sorted(:,1), moments_sorted(:,2), ...
                 moments_sorted(:,3), moments_sorted(:,4), ...
                 dist_labels_cell, ...
        'VariableNames', {'BiasedP','UnbiasedP','Mean','Variance','Skewness','Kurtosis','Distribution'});

    data = [data, param_table];

    raw_samples = samples_sorted;
    
    %% ================= SAVE TO DISK =================
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    
    % Save main data parquet (includes parameters)
    data_filename = sprintf('sign_swap_%dk_perms_n%d_%s.parquet', ...
                            round(nPermutations/1000), nParticipants, timestamp);
    fprintf('\nSaving summary data to: %s\n', data_filename);
    parquetwrite(data_filename, data);
    
    % Save raw samples to parquet
    sample_col_names = arrayfun(@(i) sprintf('S%d', i), 1:nParticipants, 'UniformOutput', false);
    samples_table = array2table(raw_samples, 'VariableNames', sample_col_names);
    samples_filename = sprintf('sign_swap_%dk_perms_n%d_%s_samples.parquet', ...
                               round(nPermutations/1000), nParticipants, timestamp);
    fprintf('Saving raw samples to: %s\n', samples_filename);
    parquetwrite(samples_filename, samples_table);

    fprintf('GPU generation complete.\n');
end


%% ============================================================================
%% PARAMETER EXTRACTION AND STATISTICS
%% ============================================================================

function param_table = extractParameterColumns(params_sorted, dist_labels_cell, dist_names)
    % Extract parameters from cell array into table columns
    %
    % Output columns use naming convention: Param_<distribution>_<parameter>
    % e.g., Param_GAMMA_shape, Param_GAMMA_scale, Param_EXPONENTIAL_rate, etc.
    
    n_samples = length(params_sorted);
    
    % Define parameter names for each distribution
    dist_param_names = struct();
    dist_param_names.GAMMA = {'shape', 'scale'};
    dist_param_names.EXPONENTIAL = {'rate'};
    dist_param_names.LOGNORMAL = {'mu', 'sigma'};
    dist_param_names.LAPLACE = {'scale'};
    dist_param_names.NORMAL = {'std'};
    dist_param_names.STUDENT_T = {'df'};
    dist_param_names.UNIFORM = {'half_width'};
    
    % Collect all unique distributions present in data
    unique_dists = unique(dist_labels_cell);
    
    % Build list of all parameter columns needed
    all_columns = {};
    for d = 1:length(unique_dists)
        dist_name = unique_dists{d};
        if isfield(dist_param_names, dist_name)
            param_names = dist_param_names.(dist_name);
            for p = 1:length(param_names)
                col_name = sprintf('Param_%s_%s', dist_name, param_names{p});
                all_columns{end+1} = col_name;
            end
        end
    end
    
    % Initialize parameter arrays with NaN
    n_cols = length(all_columns);
    param_data = nan(n_samples, n_cols);
    
    % Fill in parameter values
    for i = 1:n_samples
        dist_name = dist_labels_cell{i};
        params = params_sorted{i};
        
        if isempty(params)
            continue;
        end
        
        param_fields = fieldnames(params);
        for p = 1:length(param_fields)
            field_name = param_fields{p};
            col_name = sprintf('Param_%s_%s', dist_name, field_name);
            col_idx = find(strcmp(all_columns, col_name));
            if ~isempty(col_idx)
                param_data(i, col_idx) = params.(field_name);
            end
        end
    end
    
    % Create table
    param_table = array2table(param_data, 'VariableNames', all_columns);
end

%% ============================================================================
%% DISTRIBUTION SAMPLING
%% ============================================================================

function [sample, params] = generateDistributionSampleWithParams(n_total, dist_info)
    switch dist_info.name
        case 'GAMMA'
            shape = 0.2 + rand() * (2.0 - 0.2);
            scale = 0.5 + rand() * (4.0 - 0.5);
            sample = gamrnd(shape, scale, n_total, 1) - shape * scale;
            params = struct('shape', shape, 'scale', scale);
            
        case 'EXPONENTIAL'
            rate = 0.3 + rand() * (2.5 - 0.3);
            sample = exprnd(1/rate, n_total, 1) - 1/rate;
            params = struct('rate', rate);
            
        case 'LOGNORMAL'
            mu = -0.5 + rand() * (0.5 - (-0.5));
            sigma = 0.5 + rand() * (0.9 - 0.5);
            mean_lognormal = exp(mu + sigma^2/2);
            sample = lognrnd(mu, sigma, n_total, 1) - mean_lognormal;
            params = struct('mu', mu, 'sigma', sigma);
            
        case 'LAPLACE'
            b = 0.5 + rand() * (2.0 - 0.5);
            u = rand(n_total, 1) - 0.5;
            sample = -b * sign(u) .* log(1 - 2 * abs(u));
            params = struct('scale', b);
            
        case 'NORMAL'
            sigma = 0.5 + rand() * (2.0 - 0.5);
            sample = randn(n_total, 1) * sigma;
            params = struct('std', sigma);
            
        case 'STUDENT_T'
            df = 3 + rand() * (30 - 3);
            sample = trnd(df, n_total, 1);
            params = struct('df', df);
            
        case 'UNIFORM'
            a = 0.5 + rand() * (3.0 - 0.5);
            sample = (rand(n_total, 1) - 0.5) * 2 * a;
            params = struct('half_width', a);
            
        otherwise
            error('Unknown distribution: %s', dist_info.name);
    end
end

%% ============================================================================
%% HELPER FUNCTIONS
%% ============================================================================

function moments = calculateMoments(x)
    n = length(x);
    m = mean(x);
    x_centered = x - m;
    v = sum(x_centered.^2) / (n-1);
    
    if v > 0
        std_val = sqrt(v);
        x_standardized = x_centered / std_val;
        s = mean(x_standardized.^3) * (n/(n-1)) * (n/(n-2));
        k = mean(x_standardized.^4) - 3;
    else
        s = 0;
        k = 0;
    end
    
    moments = [m, v, s, k];
end

function p = permutation_test_one_sample(group1, permIndices)

    n = length(group1);
    nPerms = size(permIndices, 1);

    % Observed t-statistic
    mean_obs = mean(group1);
    
    std_obs = std(group1, 0);
    if std_obs == 0
        p = 1;
        return;
    end
    
    obs_t = mean_obs / (std_obs / sqrt(n));

    % Convert logicals {0,1} -> {-1,+1}
    signs = 2 .* permIndices - 1;

    % Permutation means via dot product (FAST)
    % Each row = one permutation
    perm_means = (signs * group1) ./ n;

    % Standard error of sign-flipped mean is constant
    perm_var = sum(group1.^2) ./ (n^2);
    perm_se  = sqrt(perm_var);

    % Avoid divide-by-zero
    if perm_se == 0
        t_perm = zeros(nPerms,1);
    else
        t_perm = perm_means ./ perm_se;
    end

    % Two-sided p-value
    p = (sum(abs(t_perm) >= abs(obs_t)) + 1) ./ (nPerms + 1);

end


function p = permutation_test_paired(group1, group2, swapIndices)

    n = length(group1);
    nPerms = size(swapIndices, 1);

    % Differences
    d = group1(:) - group2(:);

    % Observed t-statistic
    mean_obs = mean(d);
    se_obs = std(d) ./ sqrt(n);

    if se_obs == 0
        obs_t = 0;
    else
        obs_t = mean_obs ./ se_obs;
    end

    % Convert logical swap mask to +1 / -1 multipliers
    signs = 2 .* swapIndices - 1;

    % Permutation means (dot-product)
    perm_means = (signs * d) ./ n;

    % Standard error is constant
    perm_var = sum(d.^2) ./ (n^2);
    perm_se = sqrt(perm_var);

    if perm_se == 0
        t_perm = zeros(nPerms,1);
    else
        t_perm = perm_means ./ perm_se;
    end

    % Two-sided p-value
    p = (sum(abs(t_perm) >= abs(obs_t)) + 1) ./ (nPerms + 1);

end


function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end