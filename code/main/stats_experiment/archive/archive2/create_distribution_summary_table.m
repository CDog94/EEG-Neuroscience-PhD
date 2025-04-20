function create_distribution_summary_table(chiSquaredResults, weibullResults, lognormalResults, ...
    chiSquaredDfValues, weibullShapeValues, lognormalSigmaValues)
% Creates a summary table for the distribution testing results
%
% Parameters:
%   chiSquaredResults - Results for Chi-Squared distributions
%   weibullResults - Results for Weibull distributions
%   lognormalResults - Results for Log-normal distributions
%   chiSquaredDfValues - Array of chi-squared df values tested
%   weibullShapeValues - Array of Weibull shape parameters tested
%   lognormalSigmaValues - Array of log-normal sigma parameters tested

% Create a new figure for the summary table
figure('Position', [50, 50, 1000, 500]);  % Made wider to accommodate the additional column
set(gcf, 'Color', 'white');

% Create table data
distributionTypes = {};
parameterValues = {};
rmseValues = [];
skewnessValues = [];  % New array for skewness values

% Add Chi-Squared data
for i = 1:length(chiSquaredDfValues)
    distributionTypes{end+1} = 'Chi-Squared';
    parameterValues{end+1} = sprintf('df = %d', chiSquaredDfValues(i));
    rmseValues(end+1) = chiSquaredResults.RMSE_ByParam(i);
    
    % Calculate mean skewness for this parameter
    paramIndices = find(chiSquaredResults.Parameter == chiSquaredDfValues(i));
    meanSkewness = mean(chiSquaredResults.MeasuredSkewness(paramIndices));
    skewnessValues(end+1) = meanSkewness;
end

% Add Weibull data
for i = 1:length(weibullShapeValues)
    distributionTypes{end+1} = 'Weibull';
    parameterValues{end+1} = sprintf('shape = %.1f', weibullShapeValues(i));
    rmseValues(end+1) = weibullResults.RMSE_ByParam(i);
    
    % Calculate mean skewness for this parameter
    paramIndices = find(weibullResults.Parameter == weibullShapeValues(i));
    meanSkewness = mean(weibullResults.MeasuredSkewness(paramIndices));
    skewnessValues(end+1) = meanSkewness;
end

% Add Log-normal data
for i = 1:length(lognormalSigmaValues)
    distributionTypes{end+1} = 'Log-normal';
    parameterValues{end+1} = sprintf('sigma = %.2f', lognormalSigmaValues(i));
    rmseValues(end+1) = lognormalResults.RMSE_ByParam(i);
    
    % Calculate mean skewness for this parameter
    paramIndices = find(lognormalResults.Parameter == lognormalSigmaValues(i));
    meanSkewness = mean(lognormalResults.MeasuredSkewness(paramIndices));
    skewnessValues(end+1) = meanSkewness;
end

% Convert to column cell arrays
distributionTypes = distributionTypes';
parameterValues = parameterValues';
rmseValues = rmseValues';
skewnessValues = skewnessValues';  % Convert to column array

% Format skewness values to 4 decimal places
formattedSkewness = cellfun(@(x) sprintf('%.4f', x), num2cell(skewnessValues), 'UniformOutput', false);

% Create the table
tableData = [distributionTypes, parameterValues, formattedSkewness, num2cell(rmseValues)];
columnNames = {'Distribution Type', 'Parameter', 'Bowley''s Skewness', 'RMSE'};

% Create a uitable
t = uitable('Data', tableData, 'ColumnName', columnNames, ...
            'Position', [50 50 900 400], 'FontName', 'Arial', ...
            'ColumnWidth', {150, 150, 150, 150});
            
% Adjust the position of the table
t.Position = [50 50 900 400];

% Save the figure with the table
%saveas(gcf, 'distribution_summary_table.png');
end