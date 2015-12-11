function output_spikes = S1(TD, gabor_weight, S1_params)
%% output_spikes = S1(TD, gabor_weight, S1_params)
% Calculates the spiking S1 layer response to the Temporal Difference data
% "TD"
% TD.x -> vector of event X-addresses (in pixels)
% TD.y -> vector of event Y-addresses (in pixels)
% TD.ts -> vector of event timestamps (in microseconds)
% TD.p -> vector of event polarities (1 or -1 for ON or OFF events
% respectively)
% all fields are strictly integers only
%
% "gabor_weights" is a tensor containing all the gabor weight coefficients
% (in mV). "gabor_weights" is created using the "init_Gabor" function. An
% example can be found in the "HFIRST" function.
% 
% "S1_params" defines the neuron parameters (threshold, decay, and
% refractory period)
% 'S1_params.threshold' is threshold in mV
% 'S1_params.decay_rate' is decay rate in mV per ms
% 'S1_params.refractory_period' is the refractory period in ms
% 
% for full details see:
% Orchard, G.; Meyer, C.; Etienne-Cummings, R.; Posch, C.; Thakor, N.; and Benosman, R., "HFIRST: A Temporal Approach to Object Recognition," Pattern Analysis and Machine Intelligence, IEEE Transactions on vol.37, no.10, pp.2028-2040, Oct. 2015

if(sum(~isfield(S1_params,{'threshold', 'decay_rate', 'refractory_period'}))>0)
    error('Error: "S1_params" have not been correctly set. "S1_params" must contain the fields "threshold" (in mV), "decay_rate" (in mV per ms), and "refractory_period" (in ms)');
end
threshold           = S1_params.threshold;
decay_rate          = S1_params.decay_rate;
refractory_period   = S1_params.refractory_period;

%% convert arguments to integers and scale timestamps to milliseconds (some loss of precision)
decay_rate          = int32(decay_rate); %mV per millisecond
refractory_period   = int32(refractory_period); %milliseconds
threshold           = int32(threshold); %mV
gabor_weight        = int32(gabor_weight); %mV
TD.ts               = int32(TD.ts./1e3);

%% extract the size and indices of filters
[filter_size_y, filter_size_x, num_orientations] = size(gabor_weight);

% filter_x and filter_y are used to calculate which neurons a spike projects to
[filter_x, filter_y] = meshgrid(-floor((filter_size_x-1)/2):floor(filter_size_x/2),-floor((filter_size_y-1)/2):floor(filter_size_y/2)); 

% filter_x_index and filter_y_index are used to calculate which filter
% weights to use. The order is reversed (a:-1:1) to make this a projective
% field rather than filter
[filter_x_index, filter_y_index] = meshgrid(filter_size_x:-1:1,filter_size_y:-1:1);

%% initialize SNN state variables/holders
image_size = [max(TD.y), max(TD.x)];
number_of_events = length(TD.ts);
neuron_potential = int32(zeros([image_size, num_orientations])); % an array to hold the neuron_potentials
update_time = int32(zeros(image_size)); % an array to record when last each neuron was updated
last_spike = int32(-refractory_period*int32(ones(image_size))); % the time at which the neuron last spiked

%initialize the output spikes struct
mem_size_increment = 10000;
output_spikes.x = zeros(1,mem_size_increment);
output_spikes.y = zeros(1,mem_size_increment);
output_spikes.ts = zeros(1,mem_size_increment);
output_spikes.p = zeros(1,mem_size_increment);
output_length = mem_size_increment;

outputSpikeNumber = 1;

%% apply the filters to the data
for evtNum = 1:number_of_events             %for each event
    % find the neuron locations to update
    loc_x = TD.x(evtNum)+filter_x; %the x addresses of the neurons to update
    loc_y = TD.y(evtNum)+filter_y; %the y addresses of the neurons to update
    
    % check that these neurons lie in the image
    valid_locations = (loc_x <= image_size(2)) & (loc_y <= image_size(1)) & (loc_x > 0) & (loc_y > 0);
    loc_x = loc_x(valid_locations);
    loc_y = loc_y(valid_locations);
    filter_indices = sub2ind([filter_size_y, filter_size_x], filter_y_index(valid_locations), filter_x_index(valid_locations));
    
    % check which neurons are under refraction
    image_indices = sub2ind(image_size , loc_y(:), loc_x(:));
    filter_indices = filter_indices(TD.ts(evtNum) - last_spike(image_indices) > refractory_period);
    image_indices = image_indices(TD.ts(evtNum) - last_spike(image_indices) > refractory_period);
    
    % update the neurons
    time_since_last_update = TD.ts(evtNum) - update_time(image_indices);
    update_time(image_indices) = TD.ts(evtNum);
    neuron_indices = repmat(image_indices, 1, num_orientations) + prod(image_size).*repmat((1:num_orientations)-1, length(image_indices), 1);
    neuron_indices = neuron_indices(:);
    filter_indices = repmat(filter_indices, 1, num_orientations) + (filter_size_x*filter_size_y).*repmat((1:num_orientations)-1, length(filter_indices), 1);
    filter_indices = filter_indices(:);
    decay = min(repmat(decay_rate * time_since_last_update, num_orientations,1), abs(neuron_potential(neuron_indices))).* sign(neuron_potential(neuron_indices));
    
    neuron_potential(neuron_indices) = neuron_potential(neuron_indices) - decay;
    neuron_potential(neuron_indices) = neuron_potential(neuron_indices) + gabor_weight(filter_indices);
    
    out_spike_indices = neuron_indices(neuron_potential(neuron_indices) > threshold);
    
    if ~isempty(out_spike_indices)
        [loc_y, loc_x, orientation] = ind2sub([image_size, num_orientations], out_spike_indices);
        
        numNewEvts = numel(loc_x(:));
        outputIndices = outputSpikeNumber:(outputSpikeNumber+numNewEvts-1);
        
        if (outputSpikeNumber+numNewEvts-1) > output_length
            output_spikes.x = [output_spikes.x, zeros(1,mem_size_increment)];
            output_spikes.y = [output_spikes.y, zeros(1,mem_size_increment)];
            output_spikes.ts = [output_spikes.ts, zeros(1,mem_size_increment)];
            output_spikes.p = [output_spikes.p, zeros(1,mem_size_increment)];
            output_length = output_length + mem_size_increment;
        end
        output_spikes.x(outputIndices) = loc_x(:);
        output_spikes.y(outputIndices) = loc_y(:);
        output_spikes.ts(outputIndices) = TD.ts(evtNum)*int32(ones(size(loc_x(:))));
        output_spikes.p(outputIndices) = orientation;
        outputSpikeNumber = outputSpikeNumber + numel(loc_x(:));
        
        %record the spike time (for implementing the refractory period)
        reset_locations = sub2ind([image_size, num_orientations], loc_y, loc_x, ones(size(loc_y)));
        last_spike(reset_locations) = TD.ts(evtNum);
        
        %reset the neuron_potential
        for t = 1:length(loc_x(:))
            neuron_potential(loc_y(t), loc_x(t), :) = 0;
        end
    end
end

%% remove any extra allocated memory
output_spikes.x(outputSpikeNumber:end) = [];
output_spikes.y(outputSpikeNumber:end) = [];
output_spikes.ts(outputSpikeNumber:end) = [];
output_spikes.p(outputSpikeNumber:end) = [];