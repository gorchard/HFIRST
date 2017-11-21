# HFIRST
A Matlab implementation of the spiking HFIRST model for recognition

This Matlab code implements the model described in:
Orchard, G.; Meyer, C.; Etienne-Cummings, R.; Posch, C.; Thakor, N.; and Benosman, R., "HFIRST: A Temporal Approach to Object Recognition," Pattern Analysis and Machine Intelligence, IEEE Transactions on vol.37, no.10, pp.2028-2040, Oct. 2015 (open access arXiv link)

The code shows how to apply the model to the N-MNIST dataset described in:
Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.  “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades", Frontiers in Neuroscience, vol.9, no.437, Oct. 2015

The intention with this new Matlab code is to show how the model works. The original HMAX code was a faster C++ MEX implementation, but is far more obtuse and complex to use. The HFIRST code also relies on some basic AER Matlab functions which can be found [here](https://github.com/gorchard/Matlab_AER_vision_functions).

### This was a very early hand coded spiking model for recognition. Its accuracy and efficiency have long since been surpassed by trained models which achieve 98%+ on N-MNIST.
