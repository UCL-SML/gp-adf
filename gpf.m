function [m, S, m_t, S_t, m_y S_y] = ...
  gpf(X_t, input_t, target_t, X_o, input_o, target_o, pm, pS, y)

% Bayesian filter using GP models for transition dynamics and observation
% (trained offline)
% assumes that the GP models are NOT learned on differences
%
% inputs:
% X_t:        (D+2)*E-by-1 vector of log-hyper-parameters (transition model)
% input_t:    n-by-D matrix of training inputs (transition model)
% target_t:   n-by-E matrix of training targets (transition model)
% X_o:        (E+2)*F-by-1 vector of log-hyper-parameters (observation model)
% input_o:    n-by-E matrix of training inputs (observation model)
% target_o:   n-by-F matrix of training targets (observation model)
% pm:         D-by-1 mean of current (hidden) state distribution
% pS:         D-by-D covariance matrix of current (hidden) state distribution
% y:          F-by-1 measurement at next time step
%
% outputs:
% m:        E-by-1 mean vector of filtered distribution
% S:        E-by-E covariance matrix of filtered distribution
% m_t:      E-by-1 mean vector of the predicted state distribution
% S_t:      E-by-E covariance matrix of the predicted state distribution
% m_y:      F-by-1 mean vector of predicted measurement distribution
% S_y:      F-by-F covariance matrix of predicted measurement distribution
% 
% (C) Marc Peter Deisenroth
% 2009-07-06

% predictive state distribution p(x_t|y_1,...,y_{t-1}), no incorporation of current
% measurement
[m_t S_t] = gpPt(X_t, input_t, target_t, pm, pS); % call transition GP

% compute measurement distribution
[m_y S_y Cxy] = gpPo(X_o, input_o, target_o, m_t, S_t); % call observation GP

% filter step: combine prediction and measurement
L = chol(S_y)'; B = L\(Cxy');
m = m_t + Cxy*(S_y\(y-m_y));
S = S_t - B'*B;