% The HFIRST algorithm is described in the paper:
% Orchard, G.; Meyer, C.; Etienne-Cummings, R.; Posch, C.; Thakor, N.; and Benosman, R., "HFIRST: A Temporal Approach to Object Recognition," Pattern Analysis and Machine Intelligence, IEEE Transactions on vol.37, no.10, pp.2028-2040, Oct. 2015
% 
% On N-MNIST, this method achieves accuracy of:
% 61.34% (soft classifier) 
% 76.40% (hard classifier)
% as described in the paper:
% Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.  “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades", Frontiers in Neuroscience, vol.9, no.437, Oct. 2015
% Accuracy is slightly higher than reported in the paper due to minor
% changes in preprocessing
%  
% This file has 3 parts. It shows complete use of HFIRST training and
% testing. It assumes a directory structure where training and testing
% examples are in separate directories, with each class being a separate
% subdirectory.
% For example 'Training\class_y\file_x' where 'class_y' is a directory
% containing examples from one class only. The name of the directory can be
% anything, but should be consistent between testing and training
%
% The HFIRST function that input samples will be structs containing the
% following fields:
% TD.p is a list of polarities
% TD.x is a list of pixel X-Addresses (positive integers)
% TD.y is a list of pixel Y-Addresses (positive integers)
% TD.ts is a list of pixel timestamps in microseconds (positive integers)
%
% if using the N-MNIST or N-Caltech101 '.bin' files, then replace the
% command 'load(filename);' with 'TD = Read_Ndataset(filename);' throughout
%
% The simplified example file 'simple_example' can provide more insight into using HFIRST
% on a single sample
%
% Parameters for HFIRST itself are set inside the "HFIRST" function
%  
% There are three sections below
% Section 1 defines where the necessary input files can be found and output files will be saved
%
% Section 2 shows how to extract C1 results
%
% Section 3 shows how to generate an S2 filter from the C1 results
%
% Section 4 shows how to run HFIRST once the S2 filters have been defined
%
%% Define where inputs are located and outputs are to be saved
%define where the matlab AER functions are kept. They can be downloaded
%from: http://www.garrickorchard.com/code/matlab-AER-functions
matlab_AER_functions_location = '..\Matlab_AER_vision_functions';

%define where the NMIST dataset functions are kept (for reading and
%stabilizing). They can be downloaded with the NMIST dataset at http://www.garrickorchard.com/datasets/n-mnist
NMNIST_functions_location = '..\N-MINST';

% training_directory = 'Training'; % the root directory for the training data
training_directory = '..\..\Work\SaccadeVision\MNIST\N-MNIST\Train';

% testing_directory = 'Testing'; % the root directory for the testing data
testing_directory = '..\..\Work\SaccadeVision\MNIST\N-MNIST\Test'; % the root directory for the testing data

working_directory = 'Working'; % a directory where intermediate results can be stored. This is useful in cases where the script is interrupted

S2_filters_filename = 'S2_filters_file'; % the name of a file where the S2 filters will be stored

results_filename = 'accuracy_output'; % a file where the resulting accuracy on the test set will be saved

%% add functions to the path
addpath(genpath(matlab_AER_functions_location));
addpath(genpath(NMNIST_functions_location));
addpath(genpath('HFIRST_functions'));
if ~exist(working_directory, 'dir')
    mkdir(working_directory);
end


%% Extracting C1 outputs from training data

%create a list of training classes
training_classes = dir(training_directory);
training_classes(1:2) = [];
tic
for class_number = 1:length(training_classes) %loop through each class
    fprintf('Extracting C1 spikes for class label %i \n', class_number);
    %create a list of the filenames for this class
    filenames = dir([training_directory, '\', training_classes(class_number).name]);
    filenames(1:2) = [];
    
    if ~exist(([working_directory, '\',training_classes(class_number).name]), 'dir')
        mkdir([working_directory, '\',training_classes(class_number).name]); % create a folder to save the results
    end
    
    for filenumber = 1:length(filenames) %for each training example
        
        %load the file containing 'TD'
        TD = Read_Ndataset([training_directory, '\', training_classes(class_number).name, '\', filenames(filenumber).name]);
        TD = FilterTD(TD, 10e3); %apply some noise filtering
        %         TD = stabilize(TD); %optionally stabilize the image
        %         TD = ImplementRefraction(TD, 5e3); %optionally set a refraction time for each pixel
        
        %Run HFIRST in training mode (flag 1)
        [S1out, C1out, ~, ~] = HFIRST(TD, '', 1);
        
        %save the C1 result
        save([working_directory, '\',training_classes(class_number).name, '\', filenames(filenumber).name(1:end-4)], 'C1out');
        
        %some printing to show that progress is being made
        fprintf('.');
        if rem(filenumber, 100) == 0
            fprintf('\nProcessed 100 samples in %.2f seconds. Continuing: \n', toc);
            tic
        end
    end
end

%% Training the S2_Filters

% Set the inhibitory value assigned to synapses which do not receive a spike during
% training
S2_inhibitory_value = -1;

%create a list of training classes
training_classes = dir(working_directory);
training_classes(1:2) = [];

S2_Filter_temp = zeros(1,1,1,length(training_classes));

for class_number = 1:length(training_classes) %loop through each class
    
    %create a list of the filenames for this class
    filenames = dir([working_directory, '\', training_classes(class_number).name]);
    filenames(1:2) = [];
    
    
    for filenumber = 1:length(filenames)
        
        %load the C1 result
        load([working_directory, '\', training_classes(class_number).name, '\', filenames(filenumber).name])
        
        max_x = max(C1out.x);
        max_y = max(C1out.y);
        max_p = max(C1out.p);
        [S2FilterTemp_Sizey, S2FilterTemp_Sizex, S2FilterTemp_Sizep, ~] = size(S2_Filter_temp);
        S2_Filter_temp = padarray(S2_Filter_temp, [max(max_y - S2FilterTemp_Sizey,0), max(max_x - S2FilterTemp_Sizex,0), max(max_p - S2FilterTemp_Sizep,0)], 'post');
        
        for eventNumber = 1:length(C1out.ts)
            S2_Filter_temp(C1out.y(eventNumber), C1out.x(eventNumber), C1out.p(eventNumber), class_number) = S2_Filter_temp(C1out.y(eventNumber), C1out.x(eventNumber), C1out.p(eventNumber), class_number)+1;
        end
        fprintf('Training using class %i, sample %i, \n', class_number, filenumber)
    end
    
    S2_Filter_temp(:, :, :, class_number) = 100*S2_Filter_temp(:, :, :, class_number)./sqrt(sum(sum(sum(S2_Filter_temp(:, :, :, class_number).^2)))); %set norm to 100
end
S2_Filter_temp(S2_Filter_temp == 0) = S2_inhibitory_value;
S2_Filters = S2_Filter_temp;
save(S2_filters_filename, 'S2_Filters');


%% To extract HFIRST outputs once the S2 filters have been defined

%create a list of testing classes
testing_classes = dir(testing_directory);
testing_classes(1:2) = [];

tic
for class_number = 1:length(testing_classes) % loop through each class
    fprintf('Extracting HFIRST results for class label %i \n', class_number);
    
    %within each class, find the filenames of all the examples
    filenames = dir([testing_directory, '\', testing_classes(class_number).name]);
    filenames(1:2) = [];
    
    %set the number of test samples to use
    testSize = length(filenames);  %use all of them
    
    
    accuracy{class_number} = zeros(1, testSize); %initialize the accuracy vector
    hard_accuracy{class_number} = zeros(1, testSize); %initialize the accuracy vector
    for filenumber = 1:length(filenames)
        
        %load the file
        %         load([testing_directory, '\', testing_classes(class_number).name, '\' filenames(filenumber).name])
        TD = Read_Ndataset([testing_directory, '\', testing_classes(class_number).name, '\' filenames(filenumber).name]);
        TD = FilterTD(TD, 10e3);
        %         TD = stabilize(TD);
        %         TD = ImplementRefraction(TD, 5e3);
        
        %run HFIRST
        [S1out, C1out, S2out, C2out] = HFIRST(TD, S2_filters_filename, 0); %we only care about C2 during testing
        
        %calculate the accuracy
        if ~isempty(C2out.p)
            accuracy{class_number}(filenumber) = sum(C2out.p == class_number)./length(C2out.p); %calculate the accuracy on this example
            [~, winner] = max(hist(C2out.p, 1:length(testing_classes)));
            if winner == class_number
                hard_accuracy{class_number}(filenumber) = 1;
            end
        else
            accuracy{class_number}(filenumber) = 0; %if there are no spikes, we assign zero accuracy
        end
        fprintf('Testing: class %i, sample %i, accuracy %1.2f, average accuracy %1.2f \n', class_number, filenumber, accuracy{class_number}(filenumber), mean(accuracy{class_number}(1:filenumber)))
    end
    
    per_class_accuracy(class_number) = mean(accuracy{class_number});
    hard_per_class_accuracy(class_number) = mean(hard_accuracy{class_number});
    
    
    %some printing to show that progress is being made
    fprintf('.');
    if rem(filenumber, 100) == 0
        fprintf('\nProcessed 100 samples in %.2f seconds. Continuing: \n', toc);
        tic
    end
    
end
total_accuracy = mean(per_class_accuracy); % each class contributes equally to the accuracy, regardless of whether some classes have more samples than others
hard_total_accuracy = mean(hard_per_class_accuracy); % each class contributes equally to the accuracy, regardless of whether some classes have more samples than others
%save the results
save(results_filename, 'accuracy', 'total_accuracy', 'per_class_accuracy', 'hard_per_class_accuracy');
fprintf('category %i soft accuracy = %2.2f, hard accuracy = %2.2f \n', [(1:length(testing_classes)); (per_class_accuracy*100); (hard_per_class_accuracy*100)]);
fprintf('overall soft accuracy = %2.2f \n', total_accuracy*100);
fprintf('overall hard accuracy = %2.2f \n', hard_total_accuracy*100);