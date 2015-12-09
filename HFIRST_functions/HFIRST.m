function [S1out, C1out, S2out, C2out] = HFIRST(TD, S2_path, training)
%uncomment the line below to show the input data. Two polarities of
%events will be shown in two different colours
S1out = [];
C1out = [];
S2out = [];
C2out = [];

% ShowTD(TD)

TDneg = RemoveNulls(TD, TD.p ~= 1); %Create a new struct of OFF-events by removing all the the ON-events
TDpos = RemoveNulls(TD, TD.p == 1); %Create a new struct of ON-events by removing all the the OFF-events


%% S1

% Create the gabor filters
gabor_params.sigma = 2.8;               %standard deviation of gaussian
gabor_params.lambda = 5;                %controls period of underlying cosine relative to filter size
gabor_params.psi = 0;                   %controls phase of the underlying cosine
gabor_params.gamma = 0.3;               %controls relative x-y extent of separable gaussian
gabor_params.size = 7;                  %size in pixels of square gabor filter (must be odd)
gabor_params.num_orientations = 12;     %number of orientations to use

gabor_weights = init_Gabor(gabor_params); %run the Gabor initialization function

%set S1 parameters
S1_threshold           = 150; %mV
S1_decay_rate          = 25; %mV per millisecond
S1_refractory_period   = 5; %milliseconds

%run the S1 layer
S1pos = S1(TDpos, gabor_weights, S1_threshold, S1_decay_rate, S1_refractory_period); %process the ON-events
S1neg = S1(TDneg, gabor_weights, S1_threshold, S1_decay_rate, S1_refractory_period); %process the OFF-events
S1out = CombineStreams(S1pos, S1neg); %recombine the ON and OFF events
% S1out = CombineStreams(S1neg, S1neg); %WRONG... but better

%uncomment the line below to show the S1 data. The orientations are
%recorded as different polarities. 12 colours seen will correspond to the
%12 orientations (assuming gabor_params.num_orientations = 12). The frame rate
%is scaled because the time resolution of the spikes is now milliseconds instead of microseconds

% ShowTD(S1out, 1/24e3)

%% C1
%set the C1 parameters
C1_pooling_extent = 4; %C1 will pool over 4x4 pixel regions, non-overlapping
C1_refractory_period = 5; %refractory period for C1 in milliseconds

%run the C1 layer
C1out = S1out;
C1out.x = ceil(C1out.x/C1_pooling_extent);
C1out.y = ceil(C1out.y/C1_pooling_extent);
C1out = ImplementRefraction(C1out, C1_refractory_period, 1);

%uncomment the line below to show the C1 data. Each orientation will be a
%different colour. This should look similar to a lower spatial resolution
%version of S1. The frame rate is scaled because the time resolution of the spikes
%is now milliseconds instead of microseconds

% ShowTD(C1out, 1/24e3)

if training == 0 %training == 0 means we are testing, not training
    %% S2
    % load the S2 filters
    load(S2_path)
    
    %set S2 parameters
    S2_threshold            = 150; %mV
    S2_decay_rate           = 1; %mV per millisecond
    S2_refractory_period    = 5; %milliseconds
    S2_Filters(S2_Filters<1) = -1;

    %run the S2 layer
    S2out = S2(C1out, S2_Filters, S2_threshold, S2_decay_rate, S2_refractory_period); %process the ON-events
    
    %uncomment the line below to show the S2 data. Each class is recorded as a different polarity.
    %The number of classes will be equal to the number of colours displayed.
    %The frame rate is scaled because the time resolution of the spikes is now milliseconds instead of microseconds
    
    % ShowTD(S2out, 1/24e3)
    
    %% C2
    %set the C2 parameters
    C2_pooling_extent = 8; %C2 will pool over 8x8 pixel regions, non-overlapping
    C2_refractory_period = 5; %refractory period for C2 in milliseconds
    
    %run the C2 layer
    C2out = S2out;
    C2out.x = ceil(C2out.x/C2_pooling_extent);
    C2out.y = ceil(C2out.y/C2_pooling_extent);
    C2out = ImplementRefraction(C2out, C2_refractory_period, 1);
    
    %uncomment the line below to show the C2 data. Each orientation will be a
    %different colour. This should look similar to a lower spatial resolution
    %version of S2. The frame rate is scaled because the time resolution of the spikes
    %is now milliseconds instead of microseconds
    
    % ShowTD(C2out, 1/24e3)
end
