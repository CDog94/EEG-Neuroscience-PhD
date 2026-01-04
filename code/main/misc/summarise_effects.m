% Define the main directory
mainDir = 'C:\Users\CDoga\Documents\Research\PhD\results\frequency_domain\56_256ms\habituation\partitions'; % Change this to your main directory path

% Get a list of all subdirectories in the main directory
subDirs = dir(mainDir);
subDirs = subDirs([subDirs.isdir]); % Keep only directories
subDirs = subDirs(~ismember({subDirs.name}, {'.', '..'})); % Remove '.' and '..'

% Initialize variables to store extracted data
posclustersData = {};
dfData = [];
posPeakLevelStatsData = [];

% Loop through each subdirectory
for i = 1:length(subDirs)
    subDirPath = fullfile(mainDir, subDirs(i).name);
    
    % Check if 'stat' file exists in the current subdirectory
    statFilePath = fullfile(subDirPath, 'stat.mat');
    if exist(statFilePath, 'file')
        statData = load(statFilePath);
        
        % Extract posclusters and df from stat
        if isfield(statData, 'stat')
            stat = statData.stat;
            if isfield(stat, 'posclusters')
                posClustersStat = stat.posclusters;
                structLength = length(posClustersStat);
                newString = subDirs(i).name;
                newFieldData = repmat({newString}, structLength, 1);
                [posClustersStat.experiment] = deal(newFieldData{:});
                posclustersData = [posclustersData; posClustersStat];
            end
        end
    end
    
    % Check if 'pos_peak_level_stats_c_1' file exists in the current subdirectory
    posPeakFilePath = fullfile(subDirPath, 'pos_peak_level_stats_c_1.mat');
    if exist(posPeakFilePath, 'file')
        posPeakData = load(posPeakFilePath);
        
        % Extract the first row of data
        if isfield(posPeakData, 'pos_all_stats')
            peakStats = posPeakData.pos_all_stats(1, :);
            peakStats.experiment = subDirs(i).name;
            posPeakLevelStatsData = [posPeakLevelStatsData; peakStats];
        end
    end
end

% Convert extracted data to table format
if ~isempty(posclustersData)
    posclustersTable = struct2table(posclustersData);
else
    posclustersTable = table(); % Create an empty table if no data
end

if ~isempty(posPeakLevelStatsData)
    posPeakLevelStatsTable = array2table(posPeakLevelStatsData);
else
    posPeakLevelStatsTable = table(); % Create an empty table if no data
end

% Combine all data into one table
combinedTable = [posclustersTable, posPeakLevelStatsTable];

% Display the combined table
disp(combinedTable);