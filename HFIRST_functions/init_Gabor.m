function gabor_weight = init_Gabor(gabor_params)
% function gabor_weight = init_Gabor(gabor_params)
% returns a set of gabor filter weights 'gabor_weight'
% 
% 'gabor_weight(x,y,o)' is the weight coefficient of the gabor filter at
% location (x,y) for the filter with orientation index 'o'. 'o' is an
% integer index in range '1' to 'gabor_params.num_orientations'
%
% Filters are normalized to each have zero mean and l_2 norm of 100
% 
% Input parameters with example values are:
% gabor_params.sigma = 2.8;               %standard deviation of gaussian
% gabor_params.lambda = 5;                %controls period of underlying cosine relative to filter size
% gabor_params.psi = 0;                   %controls phase of the underlying cosine
% gabor_params.gamma = 0.3;               %controls relative x-y extent of separable gaussian
% gabor_params.size = 7;                  %size in pixels of square gabor filter (must be odd)
% gabor_params.num_orientations = 12;     %number of orientations to use
%
% For more information see equation 2 in:
% Orchard, G.; Meyer, C.; Etienne-Cummings, R.; Posch, C.; Thakor, N.; and Benosman, R., "HFIRST: A Temporal Approach to Object Recognition," Pattern Analysis and Machine Intelligence, IEEE Transactions on vol.37, no.10, pp.2028-2040, Oct. 2015
%

gabor_weight = zeros([gabor_params.size, gabor_params.size, gabor_params.num_orientations]);
for angle_index = (1:gabor_params.num_orientations)
    angle = pi/2 + pi*((angle_index-1)/gabor_params.num_orientations);
    gabor_temp = GaborFunction(gabor_params.sigma,angle,gabor_params.lambda,gabor_params.psi,gabor_params.gamma,gabor_params.size);
    
    %normalize the filter
    gabor_temp = gabor_temp - mean(gabor_temp(:)); %zero mean
    gabor_weight(:,:,angle_index) = 100*gabor_temp./sqrt(sum(gabor_temp(:).^2)); %l_2 norm of 100
end