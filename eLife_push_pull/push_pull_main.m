% Simulate the continuous model with push-pull feedback

% Author: Xiao Liu, may-10, 2020
% xiaoliu23@pku.edu.cn
% @Peking University

%% Load parameters
% setWorkPath;
addpath(fullfile('./', '2layer'));
addpath(fullfile('./', '3layer'));
addpath(fullfile('./', 'realimage'));
 

%% Perform virtual experiments
flagTask = 3;
% 1. 2layer model: 
% The neural population activity and the retrieval accuracies over time
% 2. 2layer model:
% retrieval improvement Vs the intra-class correlation strength b1
% retrieval improvement Vs the intra/inter -class noise in the external input
% 3. 2layer model:
% retrieval improvement with varying moments of applying feedbacks.
% 4. 3layer model
% 5. real image task

switch flagTask
    case 1
        continuous_performance;
    case 2
        b1_correlation;
        push_pull_lmd1;
        push_pull_lmd2;
    case 3 
        time_push;
    case 4
        main_3layer;
    case 5
        chi_dynamic;
end