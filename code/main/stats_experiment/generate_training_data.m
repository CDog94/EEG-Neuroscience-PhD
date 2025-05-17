%% Enhanced parallel QQ Plot analysis of symmetric vs nonsymmetric distributions
% This script compares p-value bias in multiple symmetric and nonsymmetric distributions
% with parallel processing for improved performance

%% Configuration
rng(42); % Set random seed for reproducibility
sampleSize = 50; % Sample size per group
nPermutations = 1000; % Number of permutations for each test
nSamples = 1000; % Number of samples to generate for each distribution

% Create output directory if it doesn't exist
outputPath = './results';
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

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
    {'Normal', 'normal', [0, 1], true}, % (mean, std)
    {'Student_t3', 't', 3, true}, % 3 degrees of freedom
    {'Student_t5', 't', 5, true}, % 5 degrees of freedom
    {'Student_t10', 't', 10, true}, % 10 degrees of freedom
    {'Uniform', 'uniform', [-0.5, 0.5], true}, % (min, max)
    {'Laplace', 'laplace', [0, 1], true}, % (location, scale)
    {'Logistic', 'logistic', [0, 1], true}, % (location, scale)
    
    % Nonsymmetric distributions
    {'ChiSquared_df1', 'chi', 1, false}, % 1 degree of freedom
    {'ChiSquared_df3', 'chi', 3, false}, % 3 degrees of freedom
    {'ChiSquared_df5', 'chi', 5, false}, % 5 degrees of freedom
    {'Exponential', 'exp', 1, false}, % rate parameter
    {'Gamma_1_1', 'gamma', [1, 1], false}, % (shape, scale)
    {'Gamma_2_1', 'gamma', [2, 1], false}, % (shape, scale)
    {'Weibull_0.5', 'weibull', 0.5, false}, % shape parameter
    {'Weibull_2', 'weibull', 2, false}, % shape parameter
    {'LogNormal_0.5', 'lognormal', 0.5, false}, % sigma parameter
    {'LogNormal_1', 'lognormal', 1, false}, % sigma parameter
    {'Beta_2_5', 'beta', [2, 5], false}, % (alpha, beta) parameters
    {'F_dist_5_2', 'f', [5, 2], false} % (df1, df2) parameters
};

nDist = length(distributions);
fprintf('Testing %d distributions (%d symmetric, %d nonsymmetric)\n', ...
    nDist, sum([distributions{:,4}]), sum(~[distributions{:,4}]));

%% Generate and test distributions
fprintf('Generating %d samples for each distribution (n=%d) with parallel processing...\n', ...
    nSamples, sampleSize);

% Initialize cell array to store results
allPValues = cell(nDist, 1);
allSkewness = cell(nDist, 1);
allRMSE = zeros(nDist, 1);

% Process each distribution
parfor distIdx = 1:nDist
    distInfo = distributions{distIdx};
    distName = distInfo{1};
    distType = distInfo{2};
    distParam = distInfo{3};
    
    fprintf('Processing %s distribution...\n', distName);
    
    % Initialize arrays for this distribution
    p_values = zeros(nSamples, 1);
    skew_values = zeros(nSamples, 1);
    
    % Group 1 is always zeros for all tests
    group1 = zeros(sampleSize, 1);
    
    % Generate samples in batches for better parallel performance
    batchSize = 100; % Process in batches of 100
    nBatches = ceil(nSamples / batchSize);
    
    for batchIdx = 1:nBatches
        % Calculate start and end indices for this batch
        startIdx = (batchIdx - 1) * batchSize + 1;
        endIdx = min(batchIdx * batchSize, nSamples);
        batchCount = endIdx - startIdx + 1;
        
        % Pre-allocate batch results
        batch_p = zeros(batchCount, 1);
        batch_skew = zeros(batchCount, 1);
        
        % Generate data for each sample in the batch
        for i = 1:batchCount
            sampleIdx = startIdx + i - 1;
            
            % Generate data based on distribution type
            switch lower(distType)
                case 'normal'
                    mean_val = distParam(1);
                    std_val = distParam(2);
                    group2 = normrnd(mean_val, std_val, sampleSize, 1) - mean_val;
                    
                case 't'
                    df = distParam;
                    group2 = trnd(df, sampleSize, 1);
                    
                case 'uniform'
                    min_val = distParam(1);
                    max_val = distParam(2);
                    group2 = (max_val - min_val) * rand(sampleSize, 1) + min_val;
                    
                case 'laplace'
                    loc = distParam(1);
                    scale = distParam(2);
                    u = rand(sampleSize, 1) - 0.5;
                    group2 = loc - scale * sign(u) .* log(1 - 2*abs(u));
                    
                case 'logistic'
                    loc = distParam(1);
                    scale = distParam(2);
                    u = rand(sampleSize, 1);
                    group2 = loc + scale * log(u ./ (1 - u));
                    
                case 'chi'
                    df = distParam;
                    mean_chi = df;
                    group2 = chi2rnd(df, sampleSize, 1) - mean_chi;
                    
                case 'exp'
                    rate = distParam;
                    mean_exp = 1/rate;
                    group2 = exprnd(mean_exp, sampleSize, 1) - mean_exp;
                    
                case 'gamma'
                    shape = distParam(1);
                    scale = distParam(2);
                    mean_gamma = shape * scale;
                    group2 = gamrnd(shape, scale, sampleSize, 1) - mean_gamma;
                    
                case 'weibull'
                    shape = distParam;
                    scale = 1.0;
                    mean_weibull = scale * gamma(1 + 1/shape);
                    group2 = wblrnd(scale, shape, sampleSize, 1) - mean_weibull;
                    
                case 'lognormal'
                    sigma = distParam;
                    mu = -sigma^2/2; % Makes E[X] = 1
                    group2 = lognrnd(mu, sigma, sampleSize, 1) - 1;
                    
                case 'beta'
                    alpha = distParam(1);
                    beta_param = distParam(2);
                    mean_beta = alpha / (alpha + beta_param);
                    group2 = betarnd(alpha, beta_param, sampleSize, 1) - mean_beta;
                    
                case 'f'
                    df1 = distParam(1);
                    df2 = distParam(2);
                    mean_f = df2 / (df2 - 2);  % Only defined for df2 > 2
                    if df2 <= 2
                        mean_f = 1; % Use approximation if mean is undefined
                    end
                    group2 = frnd(df1, df2, sampleSize, 1) - mean_f;
                    
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
    end
    
    % Sort p-values
    p_values = sort(p_values);
    
    % Calculate theoretical uniform p-values
    theoretical = (1:nSamples)' / (nSamples + 1);
    
    % Calculate RMSE (bias measure)
    rmse = sqrt(mean((p_values - theoretical).^2));
    
    % Store results
    allPValues{distIdx} = p_values;
    allSkewness{distIdx} = skew_values;
    allRMSE(distIdx) = rmse;
end

%% Calculate average skewness for each distribution
avgSkewness = zeros(nDist, 1);
for i = 1:nDist
    avgSkewness(i) = mean(allSkewness{i});
end

%% Print results
fprintf('\nResults:\n');
fprintf('%-20s %-12s %-12s %-12s\n', 'Distribution', 'Type', 'RMSE', 'Avg Skewness');
fprintf('%-20s %-12s %-12s %-12s\n', '-----------', '----', '----', '------------');

for i = 1:nDist
    distInfo = distributions{i};
    if distInfo{4}
        typeStr = 'Symmetric';
    else
        typeStr = 'Nonsymmetric';
    end
    
    fprintf('%-20s %-12s %.6f     %.3f\n', ...
        distInfo{1}, typeStr, allRMSE(i), avgSkewness(i));
end

% Compare symmetric vs nonsymmetric
isSymmetric = [distributions{:,4}];
symmetric_rmse = allRMSE(isSymmetric);
nonsymmetric_rmse = allRMSE(~isSymmetric);
mean_symmetric_rmse = mean(symmetric_rmse);
mean_nonsymmetric_rmse = mean(nonsymmetric_rmse);

fprintf('\nSummary:\n');
fprintf('Average RMSE for symmetric distributions:     %.6f\n', mean_symmetric_rmse);
fprintf('Average RMSE for nonsymmetric distributions:  %.6f\n', mean_nonsymmetric_rmse);
fprintf('Ratio (nonsymmetric/symmetric):               %.2f\n', mean_nonsymmetric_rmse/mean_symmetric_rmse);

%% Create QQ plots
% Determine optimal layout for subplots
nRows = ceil(sqrt(nDist));
nCols = ceil(nDist / nRows);

% Create figure for QQ plots
figure('Position', [100, 100, 1200, 800]);

% Create all QQ plots
for i = 1:nDist
    subplot(nRows, nCols, i);
    
    distInfo = distributions{i};
    distName = distInfo{1};
    isSymmetric = distInfo{4};
    
    % Theoretical uniform
    theoretical = (1:nSamples)' / (nSamples + 1);
    
    % Plot QQ plot
    if isSymmetric
        lineColor = [0.2, 0.6, 0.8]; % Blue for symmetric
    else
        lineColor = [0.8, 0.4, 0.2]; % Orange for nonsymmetric
    end
    
    % Plot points
    plot(theoretical, allPValues{i}, '-', 'Color', lineColor, 'LineWidth', 1.5);
    hold on;
    
    % Add diagonal reference line
    plot([0, 1], [0, 1], 'k--');
    
    % Add RMSE and skewness text
    if isSymmetric
        typeStr = 'Sym';
    else
        typeStr = 'Nonsym';
    end
    text(0.1, 0.85, sprintf('%s (%.4f)', typeStr, allRMSE(i)), ...
        'Units', 'normalized', 'FontSize', 8);
    
    % Format
    title(distName, 'FontSize', 9);
    axis square;
    grid on;
    
    % Only show axis labels on outer plots
    if i > nRows * (nCols - 1) || i == nDist
        xlabel('Expected P-values');
    end
    if mod(i-1, nCols) == 0
        ylabel('Observed P-values');
    end
    
    % Set axis limits
    xlim([0 1]);
    ylim([0 1]);
    
    % Set tick marks
    set(gca, 'XTick', 0:0.2:1);
    set(gca, 'YTick', 0:0.2:1);
    set(gca, 'FontSize', 8);
end

% Add overall title
sgtitle('P-value QQ Plots: Symmetric vs Nonsymmetric Distributions', 'FontSize', 14);

% Create legend in a separate subplot
if nRows * nCols > nDist
    subplot(nRows, nCols, nDist + 1);
    
    hold on;
    h1 = plot(NaN, NaN, '-', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 2);
    h2 = plot(NaN, NaN, '-', 'Color', [0.8, 0.4, 0.2], 'LineWidth', 2);
    h3 = plot(NaN, NaN, 'k--', 'LineWidth', 1.5);
    
    legend([h1, h2, h3], {'Symmetric', 'Nonsymmetric', 'Ideal'}, ...
        'Location', 'SouthEast', 'FontSize', 10);
    
    axis off;
end

% Save figure
saveas(gcf, fullfile(outputPath, 'pvalue_qq_plots.png'));
saveas(gcf, fullfile(outputPath, 'pvalue_qq_plots.fig'));

fprintf('\nQQ plots saved to %s\n', fullfile(outputPath, 'pvalue_qq_plots.png'));

%% Create combined QQ plot comparing best and worst cases
figure('Position', [100, 500, 800, 400]);

% Find most biased symmetric and nonsymmetric distributions
[~, maxSymIdx] = max(symmetric_rmse);
[~, maxNonsymIdx] = max(nonsymmetric_rmse);

% Find indices in the overall list
symIndices = find(isSymmetric);
nonsymIndices = find(~isSymmetric);
worstSymIdx = symIndices(maxSymIdx);
worstNonsymIdx = nonsymIndices(maxNonsymIdx);

% Plot worst symmetric
subplot(1, 2, 1);
plot(theoretical, allPValues{worstSymIdx}, '-', 'Color', [0.2, 0.6, 0.8], 'LineWidth', 2);
hold on;
plot([0, 1], [0, 1], 'k--');
title(['Most Biased Symmetric: ', distributions{worstSymIdx}{1}]);
xlabel('Expected P-values');
ylabel('Observed P-values');
text(0.1, 0.9, sprintf('RMSE: %.4f', allRMSE(worstSymIdx)), 'Units', 'normalized');
text(0.1, 0.85, sprintf('Skew: %.3f', avgSkewness(worstSymIdx)), 'Units', 'normalized');
grid on;
axis square;

% Plot worst nonsymmetric
subplot(1, 2, 2);
plot(theoretical, allPValues{worstNonsymIdx}, '-', 'Color', [0.8, 0.4, 0.2], 'LineWidth', 2);
hold on;
plot([0, 1], [0, 1], 'k--');
title(['Most Biased Nonsymmetric: ', distributions{worstNonsymIdx}{1}]);
xlabel('Expected P-values');
ylabel('Observed P-values');
text(0.1, 0.9, sprintf('RMSE: %.4f', allRMSE(worstNonsymIdx)), 'Units', 'normalized');
text(0.1, 0.85, sprintf('Skew: %.3f', avgSkewness(worstNonsymIdx)), 'Units', 'normalized');
grid on;
axis square;

% Save figure
saveas(gcf, fullfile(outputPath, 'worst_case_qq_plots.png'));

fprintf('Analysis complete.\n');