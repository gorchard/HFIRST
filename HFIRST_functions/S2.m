function output_spikes = S2(C1_result, S2_Filters, S2_params)
% Calculates the spiking S2 layer response to the C1 output data 
% C1.x -> vector of event X-addresses
% C1.y -> vector of event Y-addresses
% C1.ts -> vector of event timestamps (in milliseconds)
% C1.p -> vector of event orientation indices (element of 1:n where there are 'n' orientations)
% all fields are strictly integers only
%
% "S2_Filters" is a tensor containing the S2 filter co-efficients (in mV).
% See the section "Training the S2_Filters" in file "full_NMNIST_example.m" to see how the filters are
% created.  
% 
% "S2_params" defines the neuron parameters (threshold, decay, and
% refractory period)
% 'S2_params.threshold' is threshold in mV
% 'S2_params.decay_rate' is decay rate in mV per ms
% 'S2_params.refractory_period' is the refractory period in ms
% 
% for full details see:
% Orchard, G.; Meyer, C.; Etienne-Cummings, R.; Posch, C.; Thakor, N.; and Benosman, R., "HFIRST: A Temporal Approach to Object Recognition," Pattern Analysis and Machine Intelligence, IEEE Transactions on vol.37, no.10, pp.2028-2040, Oct. 2015

if(sum(~isfield(S2_params,{'threshold', 'decay_rate', 'refractory_period'}))>0)
    error('Error: "S2_params" have not been correctly set. "S2_params" must contain the fields "threshold" (in mV), "decay_rate" (in mV per ms), and "refractory_period" (in ms)');
end
threshold           = S2_params.threshold;
decay_rate          = S2_params.decay_rate;
refractory_period   = S2_params.refractory_period;

%% convert arguments to integers and scale timestamps to milliseconds (some loss of precision)
decay_rate          = int32(decay_rate*10); %mV per millisecond
refractory_period   = int32(refractory_period); %milliseconds
threshold           = int32(threshold*10); %mV
S2_Filters          = int32(S2_Filters*10); %mV

%% figure out how many orientations have been used, and how many classes there are
[size_y, size_x, num_orientations, num_classes] = size(S2_Filters);

%% initialize SNN state variables/holders
image_size = [max(C1_result.y), max(C1_result.x)];
if(~isempty(image_size))
    if (image_size(1) > size_y) || (image_size(2) > size_x)
        disp('Error: Mismatch between S2 Filter size and size of the example to be processed');
        return
    end
end
number_of_events = length(C1_result.ts);
neuron_potential = int32(zeros(1, num_classes)); % an array to hold the neuron_potentials
update_time = int32(0); % an array to record when last each neuron was updated
last_spike = int32(-refractory_period); % the time at which the neuron last spiked

%initialize the output spikes struct
mem_size_increment = 100;
output_spikes.x = zeros(1,mem_size_increment);
output_spikes.y = zeros(1,mem_size_increment);
output_spikes.ts = zeros(1,mem_size_increment);
output_spikes.p = zeros(1,mem_size_increment);
output_length = mem_size_increment;

outputSpikeNumber = 1;

%% apply the filters to the data

for evtNum = 1:number_of_events             %for each event
    
    if C1_result.ts(evtNum) - last_spike > refractory_period
        % check which neurons are under refraction
        update_indices  = 1:num_classes;
        
        % update the neurons
        time_since_last_update = C1_result.ts(evtNum) - update_time;
        update_time = C1_result.ts(evtNum);
        decay = min(decay_rate*time_since_last_update*int32(ones(1, num_classes)), abs(neuron_potential(1:num_classes))).* sign(neuron_potential);
        
        neuron_potential = neuron_potential - decay;
        neuron_potential = neuron_potential + squeeze(S2_Filters(C1_result.y(evtNum), C1_result.x(evtNum), C1_result.p(evtNum), :))';
        
        out_spike_indices = find(neuron_potential > threshold);
        
        if ~isempty(out_spike_indices)
            numNewEvts = length(out_spike_indices);
            outputIndices = outputSpikeNumber:(outputSpikeNumber+numNewEvts-1);
            
            if (outputSpikeNumber+numNewEvts-1) > output_length
                output_spikes.x = [output_spikes.x, zeros(1,mem_size_increment)];
                output_spikes.y = [output_spikes.y, zeros(1,mem_size_increment)];
                output_spikes.ts = [output_spikes.ts, zeros(1,mem_size_increment)];
                output_spikes.p = [output_spikes.p, zeros(1,mem_size_increment)];
                output_length = output_length + mem_size_increment;
            end
            output_spikes.x(outputIndices) = ones(1,numNewEvts);
            output_spikes.y(outputIndices) = ones(1,numNewEvts);
            output_spikes.ts(outputIndices) = C1_result.ts(evtNum)*ones(1,numNewEvts);
            output_spikes.p(outputIndices) = out_spike_indices;
            outputSpikeNumber = outputSpikeNumber + numel(numNewEvts);
            
            %record the spike time (for implementing the refractory period)
            last_spike = C1_result.ts(evtNum);
            
            %reset the neuron_potential
            neuron_potential(:) = 0;
        end
    end
end

%% remove any extra allocated memory
output_spikes.x(outputSpikeNumber:end) = [];
output_spikes.y(outputSpikeNumber:end) = [];
output_spikes.ts(outputSpikeNumber:end) = [];
output_spikes.p(outputSpikeNumber:end) = [];