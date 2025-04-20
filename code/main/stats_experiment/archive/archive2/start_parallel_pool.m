function pool = start_parallel_pool(varargin)
% Starts a parallel pool if not already started with optimized configurations
% 
% Parameters:
%   varargin - Optional parameters:
%     'NumWorkers': Number of workers (default: max available cores - 1)
%     'ThreadsPerWorker': Number of threads per worker (default: 1)
%
% Returns:
%   pool - Handle to the created parallel pool

% Process optional parameters
p = inputParser;
p.addParameter('NumWorkers', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
p.addParameter('ThreadsPerWorker', 1, @(x) isnumeric(x) && x > 0);
p.parse(varargin{:});

params = p.Results;

% Check if pool already exists
currentPool = gcp('nocreate');
if ~isempty(currentPool)
    fprintf('Using existing parallel pool with %d workers\n', currentPool.NumWorkers);
    pool = currentPool;
    return;
end

% Configure cluster
c = parcluster('local');

% Determine number of workers
if isempty(params.NumWorkers)
    % Auto-configure: use maximum cores - 1 for workers
    maxCores = feature('numcores');
    numWorkers = max(1, min(24, maxCores - 1)); % Cap at 24, leave 1 for main process
else
    numWorkers = params.NumWorkers;
end

% Set cluster configuration
c.NumWorkers = numWorkers;
saveProfile(c);

% Start the pool with optimized settings
fprintf('Starting parallel pool with %d workers...\n', numWorkers);
pool = parpool('local', numWorkers);

% Optimize thread usage
maxNumCompThreads(params.ThreadsPerWorker);
fprintf('Pool ready with %d workers, %d threads per worker\n', ...
    pool.NumWorkers, params.ThreadsPerWorker);
end