function moments = calculate_moments(data)
% Calculates the four central moments of a data distribution
% 
% Parameters:
%   data - Array of data points
%
% Returns:
%   moments - Structure containing the four moments:
%     .mean - First moment (mean)
%     .variance - Second moment (variance)
%     .skewness - Third moment (skewness)
%     .kurtosis - Fourth moment (kurtosis)

    % Calculate first moment - mean
    moments.mean = mean(data);
    
    % Calculate second moment - variance
    moments.variance = var(data);
    
    % Calculate third moment - skewness using MATLAB's built-in function
    moments.skewness = skewness(data);
    
    % Calculate fourth moment - kurtosis using MATLAB's built-in function
    % Note: MATLAB's kurtosis is excess kurtosis (normal = 0)
    moments.kurtosis = kurtosis(data);
end