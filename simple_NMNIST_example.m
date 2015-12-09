% This file has 3 parts. It shows the most basic use of HFIRST on only a single example.
%
% See 'full_example' for a more complete use incorporating multiple examples for
% training and testing
%
% Section 1 shows how to extract C1 results
%
% Section 2 shows how to generate an S2 filter from the C1 result of a
% single example (in practice you would want to use multiple examples, see
% 'full_example')
%
% Section 3 shows how to run HFIRST on an example once the S2 filter has
% been defined
%
%% To extract the C1 output for use in training
%define where the matlab AER functions are kept. They can be downloaded
%from: http://www.garrickorchard.com/code/matlab-AER-functions
matlab_AER_functions_location = '..\Matlab_AER_vision_functions';

%define where the NMIST dataset functions are kept (for reading and
%stabilizing). They can be downloaded with the NMIST dataset at http://www.garrickorchard.com/datasets/n-mnist
NMNIST_functions_location = '..\N-MINST';

%replace this name with the name of your own '.mat' file containing the TD struct.
% input_filename = 'recording';
input_filename = 'D:\Dropbox\Work\SaccadeVision\MNIST\N-MNIST\Train\0\00002.bin';

%replace this name with the name of the file where you want to store the results
output_filename = 'stage1_results';

%% add functions to the path
addpath(genpath(matlab_AER_functions_location));
addpath(genpath(NMNIST_functions_location));
addpath(genpath('HFIRST_functions'));
if ~exist(working_directory, 'dir')
    mkdir(working_directory);
end

%%
%load the file
% load(input_filename); %if a '.mat' file
TD = Read_Ndataset(input_filename); % if using an N-Caltech101 or N-MNIST '.bin' file

TD = FilterTD(TD, 10e3); % filter noise from the scene
TD = stabilize(TD); % stabilize the moving image
% TD = ImplementRefraction(TD, 5e3);

%Run HFIRST in training mode (flag 1). Parameters are set inside the HFIRST function
[~, C1out, ~, ~] = HFIRST(TD, '', 1); % In practice you would want to run the HFIRST function inside a loop, where each loop interation loads and processes one recording.

save(output_filename, 'C1out') %save the output

%% To specify the S2 filter
%replace this name with the name of the file where the C1 results are stored
output_filename = 'stage1_results';

%replace this name with the name of the file you want to use to hold the 'S2_filters' variable
filter_filename = 'filters';

%load the C1 results
load(output_filename);

%initialize a filter
S2_Filter_temp = zeros(1,1);

%figure out what the filter size should be and increase the S2 size accordingly
max_x = max(C1out.x);
max_y = max(C1out.y);
max_p = max(C1out.p);
[S2FilterTemp_Sizey, S2FilterTemp_Sizex, S2FilterTemp_Sizep, ~] = size(S2_Filter_temp);
S2_Filter_temp = padarray(S2_Filter_temp, [max(max_y - S2FilterTemp_Sizey,0), max(max_x - S2FilterTemp_Sizex,0), max(max_p - S2FilterTemp_Sizep,0)], 'post');
%count the C1 spikes at each location
for eventNumber = 1:length(C1out.ts)
    S2_Filter_temp(C1out.y(eventNumber), C1out.x(eventNumber), C1out.p(eventNumber)) = S2_Filter_temp(C1out.y(eventNumber), C1out.x(eventNumber), C1out.p(eventNumber))+1;
end

%normalize the filter
S2_Filter_temp = 100*S2_Filter_temp./sqrt(sum(sum(sum(S2_Filter_temp.^2)))); %set norm to 100
S2_Filters = S2_Filter_temp;
S2_Filters(S2_Filters == 0) = -1; %small inhibitory value
save(filter_filename, 'S2_Filters'); %save the filter

%% To extract HFIRST outputs once the S2 filters have been defined

%replace this name with the name of your own '.mat' file containing the TD struct.
% input_filename = 'recording2';
input_filename = 'D:\Dropbox\Work\SaccadeVision\MNIST\N-MNIST\Train\0\00022.bin';

%replace this name with the name of your own '.mat' file containing the 'S2_filters' variable
filter_filename = 'filters';

%replace this name with the name of the file where you want to store the results
output_filename = 'results';

%load the input data file
% load(input_filename); %if a '.mat' file
TD = Read_Ndataset(input_filename); % if using an N-Caltech101 or N-MNIST '.bin' file

TD = FilterTD(TD, 10e3);
TD = stabilize(TD);
% TD = ImplementRefraction(TD, 5e3);

%Run HFIRST in testing mode (flag 1). Parameters are set inside the HFIRST function
[S1out, C1out, S2out, C2out] = HFIRST(TD, filter_filename, 0);

save(output_filename, 'S1out', 'C1out', 'S2out', 'C2out'); %save the results

% optionally show the output
% C2out 