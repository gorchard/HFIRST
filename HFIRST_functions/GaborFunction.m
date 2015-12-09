function gb=GaborFunction(sigma,theta,lambda,psi,gamma,gabor_size)
% Name : GaborFunctionEHMAX
%
% @param sigma represent the gaussian standard deviation of gabor filter
% @param theta(rad) is the main orientation of gabor filter
% @param lambda is the cosine period of gabor filter
% @param psi is the cosine phase shift of gabor filter
% @param gamma is the ponderation factor of sigma applied in the main orientation of gabor filter
% @param gabor_size is the filter size
% @out gb is a (gabor_size)x(gabor_size) matrix representing gabor filter
%
% Description : This function returns the theta-oriented 2D gabor filter with gabor parameters, sigma, lambda, psi and gamma
%   Gabor filter gb is a (2*oriDmax+1)x(2*oriDmax+1) matrix whose elements are given by :
%
%                  1                                                xtet
% gb(x,y)=exp(- --------*( xtet^2 + gamma^2*ytet^2 ) )* cos( 2*pi* ------ +psi )
%               2sigma^2                                           lambda
%
% with xtet=x*cos(theta)+y*sin(theta)
%      ytet=-x*sin(theta)+y*cos(theta)
%
% Example : gb=GaborFunction(1,pi/4,4,0,0.2,9);surf(gb);
%
% author : Cédric MEYER
% date : 08/02/2012

theta = theta + pi/2;
oriDmax = floor(gabor_size/2);
% Bounding box
[x,y] = meshgrid(-oriDmax:oriDmax,-oriDmax:oriDmax);
% Rotation
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);


gb = exp(-1/(2*sigma^2)*(x_theta.^2+gamma^2*y_theta.^2)).*cos(2*pi/lambda*x_theta+psi);
end