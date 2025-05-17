%% Simulation to demonstrate kurtosis effect on FPR using Gamma distributions
% Gamma allows testing of non-symmetric distributions with varying kurtosis

%% Configuration
rng(42); % Set random seed for reproducibility
sampleSize = 100; % Sample size per group
nPermutations = 1000; % Number of permutations for each test
nSimulations = 300; % Number of simulations per shape value
alpha = 0.05; % Significance threshold

% Define gamma shape parameters to test (controls kurtosis)
% For gamma: kurtosis = 6/shape + 3
% Lower shape = higher kurtosis and higher skewness
shape_values = [0.5, 1, 2, 3, 5, 10, 20];
n_shapes = length(shape_values);

% Calculate theoretical kurtosis for each shape value
theoretical_kurtosis = 6./shape_values + 3;
theoretical_skewness = 2./sqrt(shape_values);

% Initialize results arrays
falsePositiveRates = zeros(n_shapes, 1);
empirical_kurtosis = zeros(n_shapes, 1);
empirical_skewness = zeros(n_shapes, 1);
confidence_intervals = zeros(n_shapes, 2);

%% Set up parallel processing
% Check if parallel pool exists, start one if not
if isempty(gcp('nocreate'))
    % Get number of available cores
    numCores = feature('numcores');
    % Use 1 fewer than available to keep system responsive
    poolSize = max(1, numCores - 1);
    fprintf('Starting parallel pool with %d workers...\n', poolSize);
    p = parpool('local', poolSize);
else
    p = gcp;
    fprintf('Using existing parallel pool with %d workers.\n', p.NumWorkers);
end

%% Run simulations
fprintf('Starting simulations with %d runs per shape value...\n', nSimulations);
totalStartTime = tic;

% Run simulations for each shape value
for shape_idx = 1:n_shapes
    shape = shape_values(shape_idx);
    scale = 1; % Fixed scale parameter = 1 for all tests
    
    % Display current shape
    fprintf('\nSimulating Gamma distribution with shape = %.1f (kurtosis ≈ %.1f, skewness ≈ %.1f)\n', ...
        shape, theoretical_kurtosis(shape_idx), theoretical_skewness(shape_idx));
    
    shapeStartTime = tic;
    
    % Storage for p-values, kurtosis, and skewness values
    p_values = zeros(nSimulations, 1);
    kurtosis_values = zeros(nSimulations, 1);
    skewness_values = zeros(nSimulations, 1);
    
    % Process in batches for better parallel performance
    batchSize = min(200, nSimulations); % Process in batches of 200 or smaller
    nBatches = ceil(nSimulations / batchSize);
    
    % Run batches in parallel
    for batchIdx = 1:nBatches
        % Calculate start and end indices for this batch
        startIdx = (batchIdx - 1) * batchSize + 1;
        endIdx = min(batchIdx * batchSize, nSimulations);
        batchCount = endIdx - startIdx + 1;
        
        % Progress reporting
        fprintf('  Processing batch %d/%d (simulations %d-%d)...\n', ...
            batchIdx, nBatches, startIdx, endIdx);
        batchStartTime = tic;
        
        % Pre-allocate arrays for batch results
        batch_p = zeros(batchCount, 1);
        batch_kurt = zeros(batchCount, 1);
        batch_skew = zeros(batchCount, 1);
        
        % Parallelize the simulations within each batch
        parfor sim_idx = 1:batchCount
            % Use fixed seeds to ensure reproducibility across parallel workers
            sim_seed = 42 + startIdx + sim_idx - 1;
            rng(sim_seed);
            
            % Generate data: Group 1 is always zeros
            group1 = zeros(sampleSize, 1);
            
            % Generate Group 2 from gamma distribution with specified shape
            % For fair comparison, we center the distribution to have mean = 0
            raw_gamma = gamrnd(shape, scale, sampleSize, 1);
            mean_gamma = shape * scale; % Expected mean of gamma(shape, scale)
            group2 = raw_gamma - mean_gamma; % Center the distribution
            
            % Store kurtosis and skewness of this sample
            batch_kurt(sim_idx) = kurtosis(group2);
            batch_skew(sim_idx) = skewness(group2);
            
            % Perform permutation test
            batch_p(sim_idx) = permutation_test(group1, group2, nPermutations);
        end
        
        % Store batch results
        p_values(startIdx:endIdx) = batch_p;
        kurtosis_values(startIdx:endIdx) = batch_kurt;
        skewness_values(startIdx:endIdx) = batch_skew;
        
        % Report batch completion
        batchTime = toc(batchStartTime);
        testsPerSecond = batchCount / batchTime;
        fprintf('  Batch completed in %.2f seconds (%.1f tests/sec)\n', ...
            batchTime, testsPerSecond);
    end
    
    % Calculate false positive rate (proportion of p < alpha)
    falsePositiveRates(shape_idx) = mean(p_values < alpha);
    
    % Calculate 95% confidence interval using binomial distribution
    [confidence_intervals(shape_idx, :)] = binofit(sum(p_values < alpha), nSimulations, 0.05);
    
    % Calculate average empirical kurtosis and skewness
    empirical_kurtosis(shape_idx) = mean(kurtosis_values);
    empirical_skewness(shape_idx) = mean(skewness_values);
    
    % Report results for this shape value
    shapeTime = toc(shapeStartTime);
    fprintf('  shape = %.1f: FPR = %.4f (95%% CI: [%.4f, %.4f]), Emp. Kurtosis = %.2f, Emp. Skewness = %.2f\n', ...
        shape, falsePositiveRates(shape_idx), ...
        confidence_intervals(shape_idx, 1), confidence_intervals(shape_idx, 2), ...
        empirical_kurtosis(shape_idx), empirical_skewness(shape_idx));
    fprintf('  Completed in %.2f seconds\n', shapeTime);
end

% Report total time taken
totalTime = toc(totalStartTime);
totalSimulations = n_shapes * nSimulations;
totalPermutations = totalSimulations * nPermutations;
fprintf('\nAll simulations completed in %.2f seconds (%.2f minutes)\n', ...
    totalTime, totalTime/60);
fprintf('Total simulations: %d\n', totalSimulations);
fprintf('Total permutation tests: %d (%.2f million)\n', ...
    totalPermutations, totalPermutations/1e6);
fprintf('Overall performance: %.1f simulations/second\n', ...
    totalSimulations/totalTime);

%% Visualize results

% Create figure
figure('Position', [100, 100, 1200, 900]);

% 1. Plot FPR vs shape parameter (controls kurtosis)
subplot(2, 3, 1);
errorbar(shape_values, falsePositiveRates, ...
    falsePositiveRates - confidence_intervals(:, 1), ...
    confidence_intervals(:, 2) - falsePositiveRates, ...
    'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
% Add reference line for alpha
plot([min(shape_values)/2, max(shape_values)*2], [alpha, alpha], 'r--', 'LineWidth', 1.5);
set(gca, 'XScale', 'log'); % Log scale for shape
xlabel('Shape Parameter (log scale)', 'FontSize', 12);
ylabel('False Positive Rate', 'FontSize', 12);
title('FPR vs. Shape Parameter', 'FontSize', 14);
grid on;
axis([min(shape_values)/1.5, max(shape_values)*1.5, 0, max([0.1, max(falsePositiveRates)*1.2])]);
% Add text annotation
text(shape_values(1), alpha + 0.01, sprintf('α = %.2f', alpha), ...
    'FontSize', 10, 'Color', 'r', 'VerticalAlignment', 'bottom');

% 2. Plot FPR vs empirical kurtosis
subplot(2, 3, 2);
plot(empirical_kurtosis, falsePositiveRates, ...
    'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
% Add reference line for alpha
plot([min(empirical_kurtosis)*0.8, max(empirical_kurtosis)*1.2], ...
    [alpha, alpha], 'r--', 'LineWidth', 1.5);
% Add regression line
p = polyfit(empirical_kurtosis, falsePositiveRates, 1);
x_fit = linspace(min(empirical_kurtosis)*0.8, max(empirical_kurtosis)*1.2, 100);
y_fit = polyval(p, x_fit);
plot(x_fit, y_fit, 'g-', 'LineWidth', 1.5);
% Add annotation with regression equation and R²
[r, pval] = corrcoef(empirical_kurtosis, falsePositiveRates);
r2 = r(1,2)^2;
text(min(empirical_kurtosis)*0.9, max(falsePositiveRates)*0.9, ...
    sprintf('y = %.4fx + %.4f\nR² = %.4f\np = %.4g', ...
    p(1), p(2), r2, pval(1,2)), ...
    'FontSize', 10, 'VerticalAlignment', 'top');
xlabel('Empirical Kurtosis', 'FontSize', 12);
ylabel('False Positive Rate', 'FontSize', 12);
title('FPR vs. Kurtosis', 'FontSize', 14);
grid on;

% 3. Plot FPR vs empirical skewness
subplot(2, 3, 3);
plot(empirical_skewness, falsePositiveRates, ...
    'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
% Add reference line for alpha
plot([min(empirical_skewness)*0.8, max(empirical_skewness)*1.2], ...
    [alpha, alpha], 'r--', 'LineWidth', 1.5);
% Add regression line
p = polyfit(empirical_skewness, falsePositiveRates, 1);
x_fit = linspace(min(empirical_skewness)*0.8, max(empirical_skewness)*1.2, 100);
y_fit = polyval(p, x_fit);
plot(x_fit, y_fit, 'g-', 'LineWidth', 1.5);
% Add annotation with regression equation and R²
[r, pval] = corrcoef(empirical_skewness, falsePositiveRates);
r2 = r(1,2)^2;
text(min(empirical_skewness)*0.9, max(falsePositiveRates)*0.9, ...
    sprintf('y = %.4fx + %.4f\nR² = %.4f\np = %.4g', ...
    p(1), p(2), r2, pval(1,2)), ...
    'FontSize', 10, 'VerticalAlignment', 'top');
xlabel('Empirical Skewness', 'FontSize', 12);
ylabel('False Positive Rate', 'FontSize', 12);
title('FPR vs. Skewness', 'FontSize', 14);
grid on;

% 4. Show examples of gamma distributions
subplot(2, 3, 4:5);
% Generate samples for visualization
n_example = 1000;
rng(42); % For reproducibility
x = linspace(0, 15, 200);

% Plot several gamma distributions
colors = jet(n_shapes);
h = zeros(n_shapes, 1);
legends = cell(n_shapes, 1);

hold on;
for i = 1:n_shapes
    shape = shape_values(i);
    gamma_samples = gamrnd(shape, scale, n_example, 1);
    
    % Plot PDF
    pdf_values = gampdf(x, shape, scale);
    h(i) = plot(x, pdf_values, 'LineWidth', 2, 'Color', colors(i,:));
    
    % Create legend entry
    legends{i} = sprintf('shape=%.1f (κ≈%.1f, γ≈%.1f)', ...
        shape, theoretical_kurtosis(i), theoretical_skewness(i));
end

xlabel('Value', 'FontSize', 12);
ylabel('Probability Density', 'FontSize', 12);
title('Gamma Distributions with Varying Shape Parameters', 'FontSize', 14);
legend(h, legends, 'Location', 'NorthEast', 'FontSize', 8);
grid on;
axis([0, 10, 0, 1]);

% 5. Table of results
subplot(2, 3, 6);
% Create a text-based table
axis off;
text_x = 0.05;
text_y = 0.95;
line_height = 0.055;

% Table header
text(text_x, text_y, 'Detailed Results:', 'FontSize', 14, 'FontWeight', 'bold');
text_y = text_y - line_height*1.5;
text(text_x, text_y, sprintf('%-6s %-8s %-8s %-8s %-8s', 'Shape', 'Kurt.', 'Skew.', 'FPR', '95% CI'), ...
    'FontSize', 12, 'FontWeight', 'bold');
text_y = text_y - line_height*0.8;
text(text_x, text_y, repmat('-', 1, 50), 'FontSize', 12);
text_y = text_y - line_height;

% Table rows
for i = 1:n_shapes
    text(text_x, text_y, sprintf('%-6.1f %-8.2f %-8.2f %-8.4f [%.4f, %.4f]', ...
        shape_values(i), empirical_kurtosis(i), empirical_skewness(i), falsePositiveRates(i), ...
        confidence_intervals(i, 1), confidence_intervals(i, 2)), 'FontSize', 10);
    text_y = text_y - line_height;
end

% Add summary statistics
text_y = text_y - line_height;
text(text_x, text_y, repmat('-', 1, 50), 'FontSize', 12);
text_y = text_y - line_height;

% Calculate multiple regression of FPR on kurtosis and skewness
X = [ones(n_shapes,1), empirical_kurtosis, empirical_skewness];
b = X\falsePositiveRates;
text(text_x, text_y, sprintf('Multiple regression:'), 'FontSize', 11, 'FontWeight', 'bold');
text_y = text_y - line_height;
text(text_x, text_y, sprintf('FPR = %.4f + %.4f·Kurt + %.4f·Skew', b(1), b(2), b(3)), ...
    'FontSize', 10);
text_y = text_y - line_height*1.5;

% Add partial correlations
text(text_x, text_y, sprintf('Correlations:'), 'FontSize', 11, 'FontWeight', 'bold');
text_y = text_y - line_height;
text(text_x, text_y, sprintf('FPR vs Kurtosis: r = %.4f', r(1,2)), 'FontSize', 10);
text_y = text_y - line_height;
[r_skew, p_skew] = corrcoef(empirical_skewness, falsePositiveRates);
text(text_x, text_y, sprintf('FPR vs Skewness: r = %.4f', r_skew(1,2)), 'FontSize', 10);
text_y = text_y - line_height;
[r_kurt_skew, p_kurt_skew] = corrcoef(empirical_kurtosis, empirical_skewness);
%% Simplified QQ Plot analysis focusing on the critical p-value region (0-0.05)
% This script compares p-value bias between symmetric and non-symmetric distributions
% with emphasis on the critical significance region

%% Configuration
% Start script timer
scriptStartTime = tic;
fprintf('Starting permutation test analysis at %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

rng(42); % Set random seed for reproducibility
sampleSize = 100; % Sample size per group
nPermutations = 5000; % Number of permutations for each test
nSamples = 3000000; % Increased number of samples for better resolution at low p-values

% Create output directory if it doesn't exist
outputPath = './results';
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
    fprintf('Created output directory: %s\n', outputPath);
end

% Report configuration settings
fprintf('\nConfiguration:\n');
fprintf('  Sample size per group: %d\n', sampleSize);
fprintf('  Number of permutations: %d\n', nPermutations);
fprintf('  Number of samples: %d\n', nSamples);
fprintf('  Total permutation tests: %d\n', nSamples * nPermutations);

%% Set up parallel processing
% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    % Get number of available cores
    numCores = feature('numcores');
    % Use 1 fewer than available to keep system responsive
    poolSize = max(1, numCores - 1);
    fprintf('Starting parallel pool with %d workers...\n', poolSize);
    parpool('local', poolSize);
else
    fprintf('Using existing parallel pool.\n');
end

%% Define distributions to test
% Format: {name, type, parameters, is_symmetric}
distributions = {
    % Symmetric distributions
    'Normal', 'normal', [0, 1], true; % (mean, std)
    'Laplace', 'laplace', [0, 1], true; % (location, scale)
    'Student_t_3', 'student_t', [3], true; % (degrees of freedom)
    'Uniform', 'uniform', [-0.5, 0.5], true; % (min, max) scaled to have var=1
    
    % Non-symmetric distributions
    'Gamma_0.5_1', 'gamma', [0.5, 1], false; % (shape=0.5, scale=1) - highly skewed
    'Exponential', 'exponential', [1], false; % (rate parameter)
    'LogNormal', 'lognormal', [0, 0.5], false; % (mu, sigma)
};

nDist = size(distributions, 1);
fprintf('Testing %d distributions with focus on critical region (p < 0.05)\n', nDist);

%% Generate and test distributions
fprintf('Generating %d samples for each distribution (n=%d) with parallel processing...\n', ...
    nSamples, sampleSize);

% Initialize progress tracking variables
overallStartTime = tic;
timeEstimates = zeros(nDist, 1);

% Initialize cell array to store results
allPValues = cell(nDist, 1);
allSkewness = cell(nDist, 1);
allRMSE = zeros(nDist, 1);
allRMSE_Critical = zeros(nDist, 1); % RMSE in critical region only

% Process each distribution
parfor distIdx = 1:nDist
    distName = distributions{distIdx, 1};
    distType = distributions{distIdx, 2};
    distParam = distributions{distIdx, 3};
    
    % Start timer for this distribution
    distStartTime = tic;
    
    fprintf('Processing %s distribution...\n', distName);
    
    % Initialize arrays for this distribution
    p_values = zeros(nSamples, 1);
    skew_values = zeros(nSamples, 1);
    
    % Group 1 is always zeros for all tests
    group1 = zeros(sampleSize, 1);
    
    % Generate samples in batches for better parallel performance
    batchSize = 100; % Process in batches of 100
    nBatches = ceil(nSamples / batchSize);
    
    % Initialize progress tracking for this distribution
    lastProgressTime = tic;
    progressUpdateInterval = 5; % Update progress every 5 seconds
    batchTimes = zeros(nBatches, 1); % Track individual batch times for more accurate estimates
    
    for batchIdx = 1:nBatches
        % Start timing this batch
        batchStartTime = tic;
        
        % Calculate start and end indices for this batch
        startIdx = (batchIdx - 1) * batchSize + 1;
        endIdx = min(batchIdx * batchSize, nSamples);
        batchCount = endIdx - startIdx + 1;
        
        % Pre-allocate batch results
        batch_p = zeros(batchCount, 1);
        batch_skew = zeros(batchCount, 1);
        
        % Progress reporting (try-catch to handle potential errors with labindex)
        try
            if labindex == 1 && toc(lastProgressTime) > progressUpdateInterval
                percentComplete = 100 * (batchIdx - 1) / nBatches;
                elapsedTime = toc(distStartTime);
                
                % More accurate remaining time estimate based on completed batch times
                if batchIdx > 1
                    completedBatchTimes = batchTimes(1:batchIdx-1);
                    validTimes = completedBatchTimes(completedBatchTimes > 0);
                    if ~isempty(validTimes)
                        avgBatchTime = mean(validTimes);
                        remainingTime = avgBatchTime * (nBatches - batchIdx + 1);
                    else
                        % Fallback if no valid times
                        estimatedTotalTime = elapsedTime / (percentComplete/100);
                        remainingTime = estimatedTotalTime - elapsedTime;
                    end
                else
                    % Initial estimate for first batch
                    estimatedTotalTime = elapsedTime * nBatches;
                    remainingTime = estimatedTotalTime - elapsedTime;
                end
                
                % Progress statistics
                fprintf('  %s: %.1f%% complete (batch %d/%d), est. %.1f min remaining\n', ...
                    distName, percentComplete, batchIdx, nBatches, remainingTime/60);
                
                lastProgressTime = tic;
            end
        catch
            % If labindex isn't available, use simpler progress reporting
            if toc(lastProgressTime) > progressUpdateInterval
                percentComplete = 100 * (batchIdx - 1) / nBatches;
                fprintf('  %s: %.1f%% complete (batch %d/%d)\n', ...
                    distName, percentComplete, batchIdx, nBatches);
                lastProgressTime = tic;
            end
        end
        
        % Generate data for each sample in the batch
        for i = 1:batchCount
            % Generate data based on distribution type
            switch lower(distType)
                case 'normal'
                    mean_val = distParam(1);
                    std_val = distParam(2);
                    group2 = normrnd(mean_val, std_val, sampleSize, 1) - mean_val;
                    
                case 'gamma'
                    shape = distParam(1);
                    scale = distParam(2);
                    mean_gamma = shape * scale;
                    group2 = gamrnd(shape, scale, sampleSize, 1) - mean_gamma;
                
                case 'laplace'
                    location = distParam(1);
                    scale = distParam(2);
                    % Generate Laplace distribution using difference of exponentials
                    u = rand(sampleSize, 1) - 0.5;
                    group2 = location - scale * sign(u) .* log(1 - 2 * abs(u));
                    
                case 'student_t'
                    df = distParam(1);
                    % Generate Student's t distribution
                    group2 = trnd(df, sampleSize, 1);
                    
                case 'uniform'
                    min_val = distParam(1);
                    max_val = distParam(2);
                    % Generate uniform distribution
                    group2 = min_val + (max_val - min_val) * rand(sampleSize, 1);
                    % No need to center as we've chosen parameters to make mean = 0
                    
                case 'exponential'
                    rate = distParam(1);
                    mean_exp = 1/rate;
                    % Generate exponential distribution
                    group2 = exprnd(1/rate, sampleSize, 1) - mean_exp;
                    
                case 'lognormal'
                    mu = distParam(1);
                    sigma = distParam(2);
                    mean_lognorm = exp(mu + sigma^2/2);
                    % Generate lognormal distribution
                    group2 = lognrnd(mu, sigma, sampleSize, 1) - mean_lognorm;
                    
                otherwise
                    error('Unknown distribution type: %s', distType);
            end
            
            % Calculate skewness
            batch_skew(i) = skewness(group2);
            
            % Perform permutation test
            batch_p(i) = permutation_test(group1, group2, nPermutations);
        end
        
        % Store batch results
        p_values(startIdx:endIdx) = batch_p;
        skew_values(startIdx:endIdx) = batch_skew;
        
        % Record batch completion time
        batchTimes(batchIdx) = toc(batchStartTime);
        
        % Every 25% completion, provide a more detailed progress update
        if mod(batchIdx, ceil(nBatches/4)) == 0
            percentComplete = 100 * batchIdx / nBatches;
            elapsedTime = toc(distStartTime);
            
            % Calculate average, min, and max batch times
            completedBatchTimes = batchTimes(1:batchIdx);
            validTimes = completedBatchTimes(completedBatchTimes > 0);
            
            if ~isempty(validTimes)
                avgBatchTime = mean(validTimes);
                minBatchTime = min(validTimes);
                maxBatchTime = max(validTimes);
                
                fprintf('\n  === %s: %.0f%% MILESTONE ===\n', distName, percentComplete);
                fprintf('    Elapsed time: %.2f minutes\n', elapsedTime/60);
                fprintf('    Batch statistics (seconds): avg=%.2f, min=%.2f, max=%.2f\n', ...
                    avgBatchTime, minBatchTime, maxBatchTime);
                
                % Estimated completion
                remainingTime = avgBatchTime * (nBatches - batchIdx);
                estimatedFinishTime = now + remainingTime/(24*60*60);
                fprintf('    Estimated completion in: %.1f minutes (around %s)\n', ...
                    remainingTime/60, datestr(estimatedFinishTime, 'HH:MM:SS'));
                fprintf('  ============================\n\n');
            end
        end
    end
    
    % Sort p-values
    p_values = sort(p_values);
    
    % Calculate theoretical uniform p-values
    theoretical = (1:nSamples)' / (nSamples + 1);
    
    % Calculate RMSE (bias measure) for all p-values
    rmse = sqrt(mean((p_values - theoretical).^2));
    
    % Calculate RMSE for critical region only (p < 0.05)
    critical_idx = theoretical <= 0.05;
    rmse_critical = sqrt(mean((p_values(critical_idx) - theoretical(critical_idx)).^2));
    
    % Store results
    allPValues{distIdx} = p_values;
    allSkewness{distIdx} = skew_values;
    allRMSE(distIdx) = rmse;
    allRMSE_Critical(distIdx) = rmse_critical;
    
    % Record time taken for this distribution
    distTime = toc(distStartTime);
    timeEstimates(distIdx) = distTime;
    
    fprintf('Completed %s distribution in %.2f minutes\n', ...
        distName, distTime/60);
end

% Report overall progress
totalTime = toc(overallStartTime);
fprintf('\nAll distributions processed in %.2f minutes (%.2f hours)\n', ...
    totalTime/60, totalTime/3600);

% Print time statistics
fprintf('\nTime Statistics:\n');
fprintf('%-20s %-15s\n', 'Distribution', 'Time (minutes)');
fprintf('%-20s %-15s\n', '-----------', '-------------');
for i = 1:nDist
    fprintf('%-20s %-15.2f\n', distributions{i, 1}, timeEstimates(i)/60);
end

% Calculate total script execution time
scriptTotalTime = toc(scriptStartTime);
fprintf('\nTotal script execution time: %.2f minutes (%.2f hours)\n', ...
    scriptTotalTime/60, scriptTotalTime/3600);

% Calculate overall performance metrics
totalPermutationTests = nDist * nSamples * nPermutations;
testsPerSecond = totalPermutationTests / scriptTotalTime;
fprintf('Performed %.2f million permutation tests\n', totalPermutationTests/1e6);
fprintf('Performance: %.2f tests per second\n', testsPerSecond);

%% Calculate average skewness for each distribution
avgSkewness = zeros(nDist, 1);
for i = 1:nDist
    avgSkewness(i) = mean(allSkewness{i});
end

%% Calculate false positive rates at alpha=0.05
falsePositiveRates = zeros(nDist, 1);
for i = 1:nDist
    % Calculate proportion of p-values below 0.05
    falsePositiveRates(i) = sum(allPValues{i} < 0.05) / nSamples;
end

%% Calculate bias ratio at alpha=0.05
% Theoretical expected count of p-values below 0.05 should be 5%
expected_sig_count = 0.05 * nSamples;
observed_sig_counts = zeros(nDist, 1);
bias_ratios = zeros(nDist, 1);

for i = 1:nDist
    observed_sig_counts(i) = sum(allPValues{i} < 0.05);
    bias_ratios(i) = observed_sig_counts(i) / expected_sig_count;
end

%% Create zoomed QQ plot focused on critical region
figure('Position', [100, 100, 1000, 800]);

% Colors for each distribution
colormap = [
    0.2, 0.6, 0.8;  % Blue (Normal)
    0.2, 0.8, 0.2;  % Green (Laplace)
    0.6, 0.2, 0.8;  % Purple (Student t)
    0.0, 0.7, 0.7;  % Cyan (Uniform)
    0.8, 0.4, 0.2;  % Orange (Gamma)
    0.8, 0.2, 0.4;  % Red (Exponential)
    0.4, 0.8, 0.2   % Lime (LogNormal)
];

% Create plot
hold on;

% Plot the ideal reference line
h_ideal = plot([0, 0.05], [0, 0.05], 'k--', 'LineWidth', 1.5);

% Find indices for the critical region
theoretical = (1:nSamples)' / (nSamples + 1);
critical_idx = theoretical <= 0.05;
theoretical_critical = theoretical(critical_idx);

% Plot each distribution in the critical region
h_dist_critical = zeros(nDist, 1);
for i = 1:nDist
    critical_values = allPValues{i}(critical_idx);
    h_dist_critical(i) = plot(theoretical_critical, critical_values, '-', 'Color', colormap(i,:), 'LineWidth', 2);
end

% Format plot
xlabel('Expected P-values', 'FontSize', 14);
ylabel('Observed P-values', 'FontSize', 14);
title('Critical Region (p < 0.05): Distribution Comparison', 'FontSize', 16);
grid on;
axis square;
axis([0 0.05 0 0.05]);

% Add more detailed tick marks for the small range
set(gca, 'XTick', 0:0.01:0.05);
set(gca, 'YTick', 0:0.01:0.05);
set(gca, 'FontSize', 12);

% Create specific legend entries for each distribution
legendStrings = cell(nDist + 1, 1);
legendStrings{1} = 'Ideal';
legendStrings{2} = sprintf('Normal (Sym, FPR=%.3f, Bias=%.1f%%)', ...
    falsePositiveRates(1), 100*(falsePositiveRates(1)-0.05)/0.05);
legendStrings{3} = sprintf('Laplace (Sym, FPR=%.3f, Bias=%.1f%%)', ...
    falsePositiveRates(2), 100*(falsePositiveRates(2)-0.05)/0.05);
legendStrings{4} = sprintf('Student t_{3} (Sym, FPR=%.3f, Bias=%.1f%%)', ...
    falsePositiveRates(3), 100*(falsePositiveRates(3)-0.05)/0.05);
legendStrings{5} = sprintf('Uniform (Sym, FPR=%.3f, Bias=%.1f%%)', ...
    falsePositiveRates(4), 100*(falsePositiveRates(4)-0.05)/0.05);
legendStrings{6} = sprintf('Gamma_{0.5} (Nonsym, FPR=%.3f, Bias=%.1f%%)', ...
    falsePositiveRates(5), 100*(falsePositiveRates(5)-0.05)/0.05);
legendStrings{7} = sprintf('Exponential (Nonsym, FPR=%.3f, Bias=%.1f%%)', ...
    falsePositiveRates(6), 100*(falsePositiveRates(6)-0.05)/0.05);
legendStrings{8} = sprintf('LogNormal (Nonsym, FPR=%.3f, Bias=%.1f%%)', ...
    falsePositiveRates(7), 100*(falsePositiveRates(7)-0.05)/0.05);

% Create legend with proper formatting
legend([h_ideal; h_dist_critical], legendStrings, 'Location', 'SouthEast', 'FontSize', 9);

% Save the critical region plot
saveas(gcf, fullfile(outputPath, 'critical_region_qq_plot.png'));
saveas(gcf, fullfile(outputPath, 'critical_region_qq_plot.fig'));