% These matlab functions contain the core of the GP-ADF-algorithm as described in
% Deisenroth, Huber, Hanebeck: Analytic Moment-based Gaussian Process Filtering, 
% International Conference on Machine Learning (ICML), 2009.
% 
% Permission is granted for anyone to copy, use, or modify this
% software and accompanying documents for any uncommercial
% purposes, provided this copyright notice is retained, and note is
% made of any changes that have been made. This software and
% documents are distributed without any warranty.
%
%
% functions included:
% 
% eps2pdf.m:   plot utility
% gpf.m:       the function that does the high-level filtering (one time step)
% gpPt.m:      GP predictions with uncertain inputs (transition dynamics)
% gpPo.m:      GP predicitons with uncertain inputs (observation function)
% gpukf.m:     GP-UKF implementation
% maha.m:      computes the pairwise squared Mahalanobis distance between to
%              sets of vectors
% scaledSymmetricSigmaPoints.m:   compute sigma points for UKF
% sim_scalar.m scalar toy example to compare filters
% trainf.m:    wrapper to train multiple-target GPs
% ukf_add.m:   UKF for additive Gaussian noise
%
%
% To run the code you need to download the GPML-software (Rasmussen and
% Williams, 2006) available at
% http://www.gaussianprocess.org/gpml/code/gpml-matlab.zip
% After downloading, add the GPML directory to the matlab path
%
% After installing the GPML package you should be able to run the GP-ADF
%
%
% example call:
% sim_scalar
%
% 
% (C) Copyright 2009, Marc Peter Deisenroth
% 
% http://mlg.eng.cam.ac.uk/marc/
% 2009-11-09
