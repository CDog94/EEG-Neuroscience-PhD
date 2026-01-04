%% Statistical Methods Calibration Analysis
% Comparing FieldTrip Monte Carlo Method vs Sign Swap Test
% Reproducing the P-P plot analysis with time filtering

clear; clc; close all;

%% Global Time Window Configuration
global TIME_WINDOW;
TIME_WINDOW = [0.056, 0.150];  % Set your desired time window here

%% Setup paths (adjust these to match your system)
main_path = 'C:\Users\CDoga\Documents\Research\PhD\participant_';
fieldtrip_path = 'C:\Users\CDoga\Documents\Research\fieldtrip-20240214';

% Add FieldTrip to path if not already added
if ~exist('ft_defaults', 'file')
    addpath(fieldtrip_path);
    ft_defaults;
end

%% Load Reference Data Structure
% Load one participant to get actual dimensions and structure
data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';

% Load one participant to get structure
fprintf('Loading reference participant data...\n');
participant_path = strcat(main_path, '1');
if ~exist(participant_path, 'dir')
    error('Could not find participant directory: %s', participant_path);
end

cd(participant_path);
if ~exist('time_domain', 'dir')
    error('Could not find time_domain directory in: %s', participant_path);
end

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

fprintf('Successfully loaded reference participant data.\n');

% Get actual dimensions from the data
n_channels = size(reference_data{1}.avg, 1);
n_timepoints = size(reference_data{1}.avg, 2);
n_participants = 20;  % As specified in your requirements

fprintf('Loaded reference data structure:\n');
fprintf('  Channels: %d\n', n_channels);
fprintf('  Time points: %d\n', n_timepoints);
fprintf('  Participants per group: %d\n', n_participants);
fprintf('  Time window: [%.3f, %.3f] s\n', TIME_WINDOW(1), TIME_WINDOW(2));

%% Simulation Parameters
n_experiments = 1;        % Number of experiments per distribution (set small for testing)
alpha = 0.05;
ft_n_permutations = 50;      % Permutations for statistical tests (set small for testing)
ss_n_permutations = 5000;

%% Distribution Parameters (all with mean=0, std=1)
distributions = struct();
distributions.NORMAL = struct('name', 'NORMAL', 'params', struct('mu', 0, 'sigma', 1));
distributions.UNIFORM = struct('name', 'UNIFORM', 'params', struct('a', -sqrt(3), 'b', sqrt(3)));
distributions.EXPONENTIAL = struct('name', 'EXPONENTIAL', 'params', struct('lambda', 1));
distributions.GAMMA = struct('name', 'GAMMA', 'params', struct('a', 4, 'b', 0.5));
distributions.LAPLACE = struct('name', 'LAPLACE', 'params', struct('mu', 0, 'b', 1/sqrt(2)));
distributions.LOGNORMAL = struct('name', 'LOGNORMAL', 'params', struct('mu', -0.5, 'sigma', 1));
distributions.STUDENT_T = struct('name', 'STUDENT_T', 'params', struct('nu', 3));

dist_names = fieldnames(distributions);

%% Initialize Results Storage
results = struct();
for i = 1:length(dist_names)
    name = dist_names{i};
    results.(name) = struct();
    results.(name).fieldtrip_pvalues = [];
    results.(name).signswap_pvalues = [];
end

%% Main Simulation Loop
fprintf('Starting Statistical Calibration Analysis...\n');
fprintf('Running %d experiments per distribution...\n', n_experiments);

for dist_idx = 1:length(dist_names)
    dist_name = dist_names{dist_idx};
    dist_info = distributions.(dist_name);
    
    fprintf('\nProcessing %s distribution...\n', dist_name);
    
    % Storage for this distribution
    fieldtrip_pvals = zeros(n_experiments, 1);
    signswap_pvals = zeros(n_experiments, 1);
    
    for exp = 1:n_experiments
        if mod(exp, 100) == 0
            fprintf('  Experiment %d/%d\n', exp, n_experiments);
        end
        
        % Generate sampled data from distribution (using reference structure)
        sampled_data = generate_sampled_data(n_participants, reference_data, dist_info);
        
        % Generate zero data for FieldTrip comparison (using reference structure)
        zero_data = generate_zero_data(n_participants, reference_data);
        
        % Run FieldTrip Monte Carlo Test (returns all p-values)
        ft_pvals = run_fieldtrip_montecarlo(zero_data, sampled_data, ft_n_permutations);
        
        % Run Sign Swap Test (returns all p-values)
        ss_pvals = run_signswap_test(sampled_data, ss_n_permutations);
        
        % Collect all p-values
        fieldtrip_pvals = [fieldtrip_pvals; ft_pvals];
        signswap_pvals = [signswap_pvals; ss_pvals];
    end
    
    % Store results
    results.(dist_name).fieldtrip_pvalues = fieldtrip_pvals;
    results.(dist_name).signswap_pvalues = signswap_pvals;
end

%% Create P-P Plots
create_qq_plots(results, alpha);


fprintf('\nAnalysis complete!\n');

%% Function Definitions

function data = generate_sampled_data(n_participants, reference_data, dist_info)
    % Generate data sampled from specified distribution using reference structure
    data = cell(n_participants, 1);
    
    % Get dimensions from reference data
    n_channels = size(reference_data{1}.avg, 1);
    n_timepoints = size(reference_data{1}.avg, 2);
    
    for p = 1:n_participants
        % Copy reference data structure
        participant_data = reference_data{1};
        
        % Generate data from specified distribution
        participant_data.avg = generate_distribution_data(n_channels, n_timepoints, dist_info);
        
        data{p} = participant_data;
    end
end

function data = generate_zero_data(n_participants, reference_data)
    % Generate zero data for control group using reference structure
    data = cell(n_participants, 1);
    
    % Get dimensions from reference data
    n_channels = size(reference_data{1}.avg, 1);
    n_timepoints = size(reference_data{1}.avg, 2);
    
    for p = 1:n_participants
        % Copy reference data structure
        participant_data = reference_data{1};
        
        % Set all values to zero
        participant_data.avg = zeros(n_channels, n_timepoints);
        
        data{p} = participant_data;
    end
end

function data_matrix = generate_distribution_data(n_channels, n_timepoints, dist_info)
    % Generate data from specified distribution
    switch dist_info.name
        case 'NORMAL'
            data_matrix = dist_info.params.mu + dist_info.params.sigma * randn(n_channels, n_timepoints);
            
        case 'UNIFORM'
            data_matrix = dist_info.params.a + (dist_info.params.b - dist_info.params.a) * rand(n_channels, n_timepoints);
            
        case 'EXPONENTIAL'
            data_matrix = exprnd(1/dist_info.params.lambda, n_channels, n_timepoints);
            % Center to mean 0
            data_matrix = data_matrix - mean(data_matrix(:));
            
        case 'GAMMA'
            data_matrix = gamrnd(dist_info.params.a, dist_info.params.b, n_channels, n_timepoints);
            % Center to mean 0
            data_matrix = data_matrix - mean(data_matrix(:));
            
        case 'LAPLACE'
            % Laplace distribution using exponential
            u = rand(n_channels, n_timepoints);
            data_matrix = dist_info.params.mu - dist_info.params.b * sign(u - 0.5) .* log(1 - 2*abs(u - 0.5));
            
        case 'LOGNORMAL'
            data_matrix = lognrnd(dist_info.params.mu, dist_info.params.sigma, n_channels, n_timepoints);
            % Center to mean 0
            data_matrix = data_matrix - mean(data_matrix(:));
            
        case 'STUDENT_T'
            data_matrix = trnd(dist_info.params.nu, n_channels, n_timepoints);
            % Normalize to unit variance
            data_matrix = data_matrix / sqrt(dist_info.params.nu/(dist_info.params.nu-2));
            
        otherwise
            error('Unknown distribution: %s', dist_info.name);
    end
end

function time_indices = get_time_indices(time_vector)
    % Get time indices for the specified time window
    global TIME_WINDOW;
    
    % Find indices within the time window
    time_indices = find(time_vector >= TIME_WINDOW(1) & time_vector <= TIME_WINDOW(2));
    
    if isempty(time_indices)
        error('No time points found in the specified time window [%.3f, %.3f]', TIME_WINDOW(1), TIME_WINDOW(2));
    end
    
    fprintf('    Time filtering: using %d time points from [%.3f, %.3f] s\n', ...
        length(time_indices), TIME_WINDOW(1), TIME_WINDOW(2));
end

function all_pvalues = run_fieldtrip_montecarlo(zero_data, sampled_data, n_permutations)
    % FieldTrip Monte Carlo method using actual ft_timelockstatistics
    global TIME_WINDOW;
    
    n_participants = length(zero_data);
    
    % Create design matrix like in original code
    design_matrix = [1:n_participants 1:n_participants; ones(1,n_participants) 2*ones(1,n_participants)];
    
    % Setup neighbours for cluster correction
    cfg = [];
    cfg.feedback = 'no';
    cfg.method = 'distance';
    cfg.elec = zero_data{1}.elec;
    neighbours = ft_prepare_neighbours(cfg);
    
    % Configure FieldTrip statistics
    cfg = [];
    cfg.latency = TIME_WINDOW;  % Use global time wi    ndow
    cfg.channel = 'eeg';
    cfg.statistic = 'ft_statfun_depsamplesT';
    cfg.method = 'montecarlo';
    cfg.correctm = 'cluster';
    cfg.neighbours = neighbours;
    cfg.clusteralpha = 0.025;
    cfg.numrandomization = n_permutations;
    cfg.tail = 0;
    cfg.design = design_matrix;
    cfg.computeprob = 'yes';
    cfg.alpha = 0.05;
    cfg.correcttail = 'alpha';
    cfg.clusterthreshold = 'nonparametric_individual';
    cfg.uvar = 1;
    cfg.ivar = 2;
    
    % Run FieldTrip statistics
    stat = ft_timelockstatistics(cfg, zero_data{:}, sampled_data{:});
    
    % Extract all p-values from the probability matrix
    all_pvalues = stat.prob(:);
end

function all_pvalues = run_signswap_test(sampled_data, n_permutations)
    % Sign Swap Test - One-sample test against zero for each channel-time point
    % Now with time filtering using the same global time window
    % Sign Swap Test - One-sample test against zero for each channel-time point
    % Simple parallel version using parfor over channels
    
    % Start parallel pool if not already running
    if isempty(gcp('nocreate'))
        % Get maximum number of processors available
        max_workers = feature('numcores');
        fprintf('    Starting parallel pool with %d workers (all available processors)...\n', max_workers);
        parpool(max_workers);
    else
        current_pool = gcp('nocreate');
        fprintf('    Using existing parallel pool with %d workers...\n', current_pool.NumWorkers);
    end
    
    n_participants = length(sampled_data);
    n_channels = size(sampled_data{1}.avg, 1);
    
    % Get time indices for filtering - use the time vector from the data structure
    time_vector = sampled_data{1}.time;
    time_indices = get_time_indices(time_vector);
    n_timepoints_filtered = length(time_indices);
    
    % Extract data: participants × channels × timepoints (time-filtered)
    group_data = zeros(n_participants, n_channels, n_timepoints_filtered);
    for p = 1:n_participants
        % Extract only the time points within the specified window
        group_data(p, :, :) = sampled_data{p}.avg(:, time_indices);
    end
    
    % Initialize p-values matrix
    all_pvalues = zeros(n_channels, n_timepoints_filtered);
    
    fprintf('    Running Sign Swap test on %d channel-time points (parallel over channels)...\n', ...
        n_channels * n_timepoints_filtered);
    
    % Parallelize over channels
    parfor ch = 1:n_channels
        if mod(ch, 10) == 0
            fprintf('      Processing channel %d/%d\n', ch, n_channels);
        end
        
        % Initialize column for this channel
        channel_pvalues = zeros(n_timepoints_filtered, 1);
        
        for tp = 1:n_timepoints_filtered
            % Get the 20 participant values at this specific channel-time point
            participant_values = group_data(:, ch, tp);  % 20×1 vector
            
            % Calculate observed t-statistic for this location
            mean_val = mean(participant_values);
            std_val = std(participant_values);
            t_observed = sqrt(n_participants) * mean_val / std_val;
            
            % Sign swap permutation test for this specific location
            null_t = zeros(n_permutations, 1);
            for perm = 1:n_permutations
                % Sign flipping for these 20 values
                signs = 2 * (rand(n_participants, 1) > 0.5) - 1;
                perm_values = signs .* participant_values;
                
                % Compute permuted t-statistic
                mean_perm = mean(perm_values);
                std_perm = std(perm_values);
                t_perm = sqrt(n_participants) * mean_perm / std_perm;
                
                null_t(perm) = t_perm;
            end
            
            % Calculate p-value for this channel-time point (two-tailed)
            p = mean(abs(null_t) >= abs(t_observed));
            channel_pvalues(tp) = p;
        end
        
        % Store results for this channel
        all_pvalues(ch, :) = channel_pvalues;
    end
    
    % Return as column vector
    all_pvalues = all_pvalues(:);
end

function create_qq_plots(results, alpha)
    % Create Q-Q plots comparing expected vs observed p-values
    % Simple version: if FPR=0.05, curve should be on diagonal
    global TIME_WINDOW;
    
    dist_names = fieldnames(results);
    
    % Colors for each distribution
    colors = [
        0.0, 0.0, 1.0;  % Blue - NORMAL
        0.0, 0.8, 0.8;  % Cyan - UNIFORM  
        1.0, 0.5, 0.0;  % Orange - EXPONENTIAL
        1.0, 0.0, 0.0;  % Red - GAMMA
        0.0, 0.8, 0.0;  % Green - LAPLACE
        0.6, 0.0, 0.8;  % Purple - LOGNORMAL
        1.0, 0.0, 1.0;  % Magenta - STUDENT_T
    ];
    
    % Create figure with subplots
    figure('Position', [100, 100, 1200, 500]);
    
    % Plot 1: FieldTrip Monte Carlo Method
    subplot(1, 2, 1);
    hold on;
    
    legend_entries = {};
    fpr_values_ft = [];
    
    for i = 1:length(dist_names)
        dist_name = dist_names{i};
        pvals = results.(dist_name).fieldtrip_pvalues;
        
        % Remove any NaN or invalid p-values
        pvals = pvals(~isnan(pvals) & pvals >= 0 & pvals <= 1);
        
        if length(pvals) > 0
            % Sort observed p-values
            sorted_pvals = sort(pvals);
            n_total = length(sorted_pvals);
            
            % Expected p-values: uniform quantiles from 0 to 1
            expected_uniform = ((1:n_total) - 0.5) / n_total;
            
            % Plot the entire curve, but focus on region [0, alpha]
            plot(expected_uniform, sorted_pvals, 'Color', colors(i, :), 'LineWidth', 2);
            
            % Calculate False Positive Rate
            fpr = mean(pvals <= alpha);
            fpr_values_ft = [fpr_values_ft, fpr];
            
            legend_entries{end+1} = sprintf('%s (FPR=%.3f)', dist_name, fpr);
        else
            fpr_values_ft = [fpr_values_ft, 0];
            legend_entries{end+1} = sprintf('%s (FPR=%.3f)', dist_name, 0);
        end
    end
    
    % Perfect calibration line (diagonal)
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 2);
    legend_entries = [{'Perfect Calibration'}, legend_entries];
    
    xlabel('Expected P-values (Uniform)');
    ylabel('Observed P-values');
    title(sprintf('FieldTrip Monte Carlo Method - Time Window [%.3f, %.3f]s', TIME_WINDOW(1), TIME_WINDOW(2)));
    legend(legend_entries, 'Location', 'northwest');
    grid on;
    %axis([0, alpha, 0, alpha]);  % Zoom to critical region
    
    % Plot 2: Sign Swap Test
    subplot(1, 2, 2);
    hold on;
    
    legend_entries = {};
    fpr_values_ss = [];
    
    for i = 1:length(dist_names)
        dist_name = dist_names{i};
        pvals = results.(dist_name).signswap_pvalues;
        
        % Remove any NaN or invalid p-values
        pvals = pvals(~isnan(pvals) & pvals >= 0 & pvals <= 1);
        
        if length(pvals) > 0
            % Sort observed p-values
            sorted_pvals = sort(pvals);
            n_total = length(sorted_pvals);
            
            % Expected p-values: uniform quantiles from 0 to 1
            expected_uniform = ((1:n_total) - 0.5) / n_total;
            
            % Plot the entire curve, but focus on region [0, alpha]
            plot(expected_uniform, sorted_pvals, 'Color', colors(i, :), 'LineWidth', 2);
            
            % Calculate False Positive Rate
            fpr = mean(pvals <= alpha);
            fpr_values_ss = [fpr_values_ss, fpr];
            
            legend_entries{end+1} = sprintf('%s (FPR=%.3f)', dist_name, fpr);
        else
            fpr_values_ss = [fpr_values_ss, 0];
            legend_entries{end+1} = sprintf('%s (FPR=%.3f)', dist_name, 0);
        end
    end
    
    % Perfect calibration line (diagonal)
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 2);
    legend_entries = [{'Perfect Calibration'}, legend_entries];
    
    xlabel('Expected P-values (Uniform)');
    ylabel('Observed P-values');
    title(sprintf('Sign Swap Test - Time Window [%.3f, %.3f]s', TIME_WINDOW(1), TIME_WINDOW(2)));
    legend(legend_entries, 'Location', 'northwest');
    grid on;
   % axis([0, alpha, 0, alpha]);  % Zoom to critical region
    
    % Add main title
    sgtitle(sprintf('Comparison of Statistical Methods - Time Window [%.3f, %.3f]s', TIME_WINDOW(1), TIME_WINDOW(2)), ...
        'FontSize', 16, 'FontWeight', 'bold');
    
    % Print summary
    fprintf('\n=== FALSE POSITIVE RATES ===\n');
    fprintf('Time Window: [%.3f, %.3f]s\n', TIME_WINDOW(1), TIME_WINDOW(2));
    fprintf('Distribution\t\tFieldTrip\tSign Swap\n');
    fprintf('----------------------------------------\n');
    for i = 1:length(dist_names)
        if i <= length(fpr_values_ft) && i <= length(fpr_values_ss)
            fprintf('%s\t\t%.3f\t\t%.3f\n', dist_names{i}, fpr_values_ft(i), fpr_values_ss(i));
        end
    end
end