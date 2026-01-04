function cfg = config_pipeline()
    % CONFIGURATION FOR P-VALUE CORRECTION PIPELINE
    % Returns struct with all parameters for data, model, training, evaluation
    
    %% Data Parameters
    cfg.data.file = 'sign_swap_40k_perms_n40_2026-01-03_14-54-10.parquet';
    cfg.data.load_existing = true;
    cfg.data.test_type = 'one-sample';  % 'one-sample' or 'paired'
    
    % Generation parameters (if not loading existing)
    cfg.data.n_pvalues = 100000;
    %cfg.data.n_pvalues = 30000;
    cfg.data.n_participants = 40;
    cfg.data.n_permutations = 10000;

    %% Filtering Parameters
    % GAMMA
    %cfg.filter.distributions_to_exclude = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T', 'LOGNORMAL', 'EXPONENTIAL'};
    % LOGNORMAL
    %cfg.filter.distributions_to_exclude = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T', 'GAMMA', 'EXPONENTIAL'};
    % EXPONENTIAL
    %cfg.filter.distributions_to_exclude = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T', 'GAMMA', 'LOGNORMAL'};
    % ALL THREE
    cfg.filter.distributions_to_exclude = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T'};

    
    cfg.filter.skewness_threshold = Inf;  % Set to Inf to disable
    
    %% generate diagnostics
    cfg.generate_diagnostics = true;  % Enable diagnostic plots

    %% Data Augmentation
    cfg.augment.use_theoretical_moments = false;
    cfg.augment.oversample.enabled = false;
    cfg.augment.oversample.range_min = 0.001;
    cfg.augment.oversample.range_max = 0.04;
    cfg.augment.oversample.factor = 3.0;
    
    %% Model Configuration
    % Model type: 'random_forest', 'svr', 'gbm', etc.
    cfg.model.type = 'random_forest';
    
    % Polynomial params
    %cfg.model.params = 99;
    
    % Random Forest parameters
    cfg.model.params.num_trees = 100;
    cfg.model.params.min_leaf_size = 2;
    cfg.model.params.mtry = 3;
    
    % Support Vector Regression parameters
     %cfg.model.params.kernel_function = 'linear';  % 'linear', 'gaussian', 'polynomial'
     %cfg.model.params.box_constraint = 1;
     %cfg.model.params.epsilon = 0.01;
     %cfg.model.params.kernel_scale = 'auto';
     %cfg.model.params.standardize = true;

    %% Training Configuration
    cfg.training.use_calibration = false;
    cfg.training.training_sample_fraction = 1;  % Use 100% of training data
    
    % Train/test split ratios
    cfg.training.train_pct = 0.8;
    cfg.training.cal_pct = 0.0;   % Set to 0.2 if using calibration
    cfg.training.test_pct = 0.2;
    
    %% Range-Specific Training
    cfg.range_specific.enabled = true;
    cfg.range_specific.training_ranges = {[0.00, 0.025], [0.025, 0.06], [0.06, 1]};  % Single range = traditional training
    % Examples:
    %   {[0, 0.04], [0.04, 0.06], [0.06, 1]}  % Multiple ranges
    %   {[0.04, 0.06]}                         % Single focused range
    cfg.range_specific.out_of_range_strategy = 'nearest';  % 'nearest' or 'passthrough'
    
    %% Calibration Parameters
    cfg.calibration.range_min = 0.04;
    cfg.calibration.range_max = 0.10;
    
    %% Evaluation Parameters
    cfg.eval.alpha = 0.05;  % Significance threshold
    cfg.eval.plot_sample_fraction = 1;  % Use 5% of test data for plotting
    cfg.eval.save_plots = true;
    
    %% Random Seed
    cfg.random_seed = 42;
    
    %% Display configuration summary
    fprintf('\n========== PIPELINE CONFIGURATION ==========\n');
    fprintf('Data file: %s\n', cfg.data.file);
    fprintf('Model type: %s\n', cfg.model.type);
    fprintf('Calibration: %s\n', ternary(cfg.training.use_calibration, 'ENABLED', 'DISABLED'));
    fprintf('Range-specific: %s\n', ternary(cfg.range_specific.enabled, 'ENABLED', 'DISABLED'));
    fprintf('Excluded distributions: %s\n', strjoin(cfg.filter.distributions_to_exclude, ', '));
    fprintf('==========================================\n\n');
end

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end