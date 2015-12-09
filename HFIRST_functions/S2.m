function output_spikes = S2(C1input, S2_Filters, threshold, decay_rate, refractory_period)
%% describe the function here
%should actually split this into two... one part for generating the
%filters, and another for applying them

%% convert arguments to integers and scale timestamps to milliseconds (some loss of precision)
decay_rate          = int32(decay_rate*10); %mV per millisecond
refractory_period   = int32(refractory_period); %milliseconds
threshold           = int32(threshold*10); %mV
S2_Filters          = int32(S2_Filters*10); %mV

%% figure out how many orientations have been used, and how many classes there are
[size_y, size_x, num_orientations, num_classes] = size(S2_Filters);

%% initialize SNN state variables/holders
image_size = [max(C1input.y), max(C1input.x)];
if(~isempty(image_size))
    if (image_size(1) > size_y) || (image_size(2) > size_x)
        disp('Error: Mismatch between S2 Filter size and size of the example to be processed');
        return
    end
end
number_of_events = length(C1input.ts);
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
    
    if C1input.ts(evtNum) - last_spike > refractory_period
        % check which neurons are under refraction
        update_indices  = 1:num_classes;
        
        % update the neurons
        time_since_last_update = C1input.ts(evtNum) - update_time;
        update_time = C1input.ts(evtNum);
        decay = min(decay_rate*time_since_last_update*int32(ones(1, num_classes)), abs(neuron_potential(1:num_classes))).* sign(neuron_potential);
        
        neuron_potential = neuron_potential - decay;
        neuron_potential = neuron_potential + squeeze(S2_Filters(C1input.y(evtNum), C1input.x(evtNum), C1input.p(evtNum), :))';
        
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
            output_spikes.ts(outputIndices) = C1input.ts(evtNum)*ones(1,numNewEvts);
            output_spikes.p(outputIndices) = out_spike_indices;
            outputSpikeNumber = outputSpikeNumber + numel(numNewEvts);
            
            %record the spike time (for implementing the refractory period)
            last_spike = C1input.ts(evtNum);
            
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