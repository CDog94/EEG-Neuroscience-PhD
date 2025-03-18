%% Main simulation script for Gamma and Uniform distributions (shifted to mean = 0)
clear; close all;

%% ===== User Settings =====
nExperiments = 5000;  % Number of experiments per parameter value  
n = 50;               % Sample size per group
nPermutations = 5000; % Number of permutations in the permutation test

% Parameter values for different distributions
% For Gamma: fix a shape parameter and loop over scale values
fixedShape = 2;             % Fixed shape parameter (a)
paramValues = 0.2:0.5:4.0;  % These will be used as scale (θ) values for gamma
                            % and half-width values for uniform

% Settings for p-value histogram: bins of width 0.01 from 0 to 1
edges = 0:0.01:1;
binCenters = edges(1:end-1) + diff(edges)/2;

% Colors for plotting (one per parameter value):
colors = lines(length(paramValues));

%% --- Arrays to store results ---
gammaPValues = cell(length(paramValues), 1);
uniformPValues = cell(length(paramValues), 1);
gammaGroup2Data = cell(length(paramValues), 1);
uniformGroup2Data = cell(length(paramValues), 1);

% Start parallel pool if not already started
if isempty(gcp('nocreate'))
    parpool;
end

% Get the number of workers
poolObj = gcp('nocreate');
numWorkers = poolObj.NumWorkers;
fprintf('Using %d parallel workers\n', numWorkers);

%% Create a progress monitoring system
progressFile = 'progress_data.txt';
if exist(progressFile, 'file')
    delete(progressFile);
end

% Write initial progress data
fid = fopen(progressFile, 'w');
fprintf(fid, '0\n');
fclose(fid);

% Create a waitbar instead of uiprogressdlg
h = waitbar(0, 'Starting simulation...', 'Name', 'Simulation Progress');

totalSimulations = length(paramValues) * nExperiments;
startTime = tic;

% Initialize a progress counter for the waitbar
pctDone = 0;

% Create a figure for plotting histograms later
plotFig = figure('Position', [100, 100, 1200, 500]);

%% Run simulations for both distributions in parallel using parfor
parfor idx = 1:length(paramValues)
    paramVal = paramValues(idx);
    
    % Arrays to store results for this parameter value
    gammaPVals = zeros(nExperiments, 1);
    uniformPVals = zeros(nExperiments, 1);
    gammaG2Data = cell(nExperiments, 1);
    uniformG2Data = cell(nExperiments, 1);
    
    % Log start of this parameter value
    fprintf('Starting parameter value %.1f (%d of %d)\n', paramVal, idx, length(paramValues));
    
    for iExp = 1:nExperiments
        % Group 1: all zeros for both distributions
        group1 = zeros(n, 1);
        
        % --- Gamma Distribution ---
        % Generate gamma data with fixed shape and scale = paramVal
        % Shift by subtracting the mean so that the data have mean = 0
        centerVal = fixedShape * paramVal; % mean of gamma distribution
        gammaGroup2 = gamrnd(fixedShape, paramVal, [n, 1]) - centerVal;
        
        % Store Group 2 data
        gammaG2Data{iExp} = gammaGroup2;
        
        % Compute the two-tailed p-value using a permutation test
        gammaPVals(iExp) = permutationMeanTest(group1, gammaGroup2, nPermutations);
        
        % --- Uniform Distribution ---
        % Generate uniform data with range [-paramVal, paramVal] (centered at 0)
        uniformGroup2 = unifrnd(-paramVal, paramVal, [n, 1]);
        
        % Store Group 2 data
        uniformG2Data{iExp} = uniformGroup2;
        
        % Compute the two-tailed p-value using a permutation test
        uniformPVals(iExp) = permutationMeanTest(group1, uniformGroup2, nPermutations);
        
        % Log progress every 500 experiments
        if mod(iExp, 500) == 0
            fprintf('  Worker %d: Parameter %.1f - Completed %d/%d experiments\n', ...
                labindex, paramVal, iExp, nExperiments);
        end
    end
    
    % Store results for this parameter value
    gammaPValues{idx} = gammaPVals;
    uniformPValues{idx} = uniformPVals;
    gammaGroup2Data{idx} = gammaG2Data;
    uniformGroup2Data{idx} = uniformG2Data;
    
    % Log completion of this parameter value
    fprintf('COMPLETED parameter value %.1f (%d of %d)\n', paramVal, idx, length(paramValues));
    
    % Signal completion by writing to a temporary file
    tmpFile = sprintf('param_done_%d.txt', idx);
    fid = fopen(tmpFile, 'w');
    fprintf(fid, '1');
    fclose(fid);
end

% Count completed parameter values and update progress bar
numCompleted = 0;
for idx = 1:length(paramValues)
    tmpFile = sprintf('param_done_%d.txt', idx);
    if exist(tmpFile, 'file')
        numCompleted = numCompleted + 1;
        delete(tmpFile);  % Clean up
    end
end

% Update waitbar
waitbar(numCompleted/length(paramValues), h, ...
    sprintf('Completed %d/%d parameter values', numCompleted, length(paramValues)));

% Display elapsed time
elapsedTime = toc(startTime);
fprintf('Total simulation time: %.2f seconds\n', elapsedTime);

% Close the waitbar
close(h);

% Create a new waitbar for plotting
h2 = waitbar(0, 'Plotting results...', 'Name', 'Plotting Progress');

% Make plotFig the current figure
figure(plotFig);

%% Pre-calculate the y-axis limits for both plots
% Calculate maximum bin count across all parameter values for both distributions
maxGammaCounts = zeros(length(paramValues), 1);
maxUniformCounts = zeros(length(paramValues), 1);

for idx = 1:length(paramValues)
    % Gamma distribution
    pvalues = gammaPValues{idx};
    counts = histcounts(pvalues, edges);
    maxGammaCounts(idx) = max(counts);
    
    % Uniform distribution
    pvalues = uniformPValues{idx};
    counts = histcounts(pvalues, edges);
    maxUniformCounts(idx) = max(counts);
end

% Set a common y-axis limit for both plots
yAxisMax = max([maxGammaCounts; maxUniformCounts]) * 1.05; % Add 5% padding

%% --- Plot 1: Gamma Distribution p-values ---
subplot(1, 2, 1);
hold on;
gammaLegendEntries = strings(length(paramValues), 1);

for idx = 1:length(paramValues)
    % Update progress bar
    waitbar(idx/(2*length(paramValues)), h2, ...
        sprintf('Plotting Gamma results %d/%d...', idx, length(paramValues)));
    
    paramVal = paramValues(idx);
    pvalues = gammaPValues{idx};
    
    % Compute histogram counts for the p-values using the specified bins
    counts = histcounts(pvalues, edges);
    
    % Plot the histogram as a bar plot
    bar(binCenters, counts, 'FaceColor', colors(idx, :), 'FaceAlpha', 0.5, 'EdgeAlpha', 0.6);
    
    % Create legend entry
    gammaLegendEntries(idx) = sprintf('Scale = %.1f, Shape = %.1f', paramVal, fixedShape);
end

xlabel('p-value');
ylabel('Frequency');
title('Histogram of p-values (Gamma Distribution)');
legend(gammaLegendEntries, 'Location', 'northwest');
grid on;
ylim([0, yAxisMax]);  % Set the y-axis limit
hold off;

%% --- Plot 2: Uniform Distribution p-values ---
subplot(1, 2, 2);
hold on;
uniformLegendEntries = strings(length(paramValues), 1);

for idx = 1:length(paramValues)
    % Update progress bar
    waitbar(0.5 + idx/(2*length(paramValues)), h2, ...
        sprintf('Plotting Uniform results %d/%d...', idx, length(paramValues)));
    
    paramVal = paramValues(idx);
    pvalues = uniformPValues{idx};
    
    % Compute histogram counts for the p-values using the specified bins
    counts = histcounts(pvalues, edges);
    
    % Plot the histogram as a bar plot
    bar(binCenters, counts, 'FaceColor', colors(idx, :), 'FaceAlpha', 0.5, 'EdgeAlpha', 0.6);
    
    % Create legend entry
    uniformLegendEntries(idx) = sprintf('Half-width = %.1f', paramVal);
end

xlabel('p-value');
ylabel('Frequency');
title('Histogram of p-values (Uniform Distribution)');
legend(uniformLegendEntries, 'Location', 'northwest');
grid on;
ylim([0, yAxisMax]);  % Set the y-axis limit with the same value as the first plot
hold off;

% Add a common title to the figure
sgtitle('P-value Distributions: Gamma vs Uniform');

% Close the progress bar
close(h2);

% Cleanup temporary files
if exist('progress_data.txt', 'file'), delete('progress_data.txt'); end

fprintf('Analysis complete! Elapsed time: %.2f seconds\n', elapsedTime);

function p = permutationMeanTest(group1, group2, nPermutations)
    % Calculate the observed difference of means.
    obsDiff = mean(group1) - mean(group2);
    
    % Combine both groups.
    combined = [group1; group2];
    n = length(group1);  % assumes equal group sizes
    
    % Preallocate an array for permutation differences.
    permDiff = zeros(nPermutations, 1);
    
    for j = 1:nPermutations
        % Randomly permute the combined data.
        permutedIdx = randperm(length(combined));
        
        % Divide the permuted data into two groups.
        permGroup1 = combined(permutedIdx(1:n));
        permGroup2 = combined(permutedIdx(n+1:end));
        
        % Compute the difference of means for this permutation.
        permDiff(j) = mean(permGroup1) - mean(permGroup2);
    end
    
    % The two-tailed p-value is the fraction of permuted differences (in absolute value)
    % that are at least as large as the observed absolute difference.
    p = mean(abs(permDiff) >= abs(obsDiff));
end