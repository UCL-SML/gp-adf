function [xEst,PEst,xPred,PPred,zPred,S,J]=...
    ukf_add(xEst,PEst,U,Q,ffun,z,R,hfun,dt,alpha,beta,kappa)

% TITLE    :  UNSCENTED KALMAN FILTER for additive noise
%
% PURPOSE  :  This function performs one complete step of the unscented Kalman filter.
%
% SYNTAX   :  [xEst,PEst,xPred,PPred,zPred,inovation]=ukf(xEst,PEst,U,Q,ffun,z,R,hfun,dt,alpha,beta,kappa)
%
% INPUTS   :  - xEst             : state mean estimate at time k  
%             - PEst             : state covariance at time k
%             - U                : vector of control inputs
%             - Q                : process noise covariance at time k
%             - ffun             : process model function  
%             - z                : observation at k+1  
%             - R                : measurement noise covariance at k+1  
%             - hfun             : observation model function  
%             - dt               : time step (passed to ffun/hfun)   
%             - alpha (optional) : sigma point scaling parameter. Defaults to 1.
%             - beta  (optional) : higher order error scaling parameter. Default to 0.  
%             - kappa (optional) : scalar tuning parameter 1. Defaults to 0.  
%
% OUTPUTS  :  - xEst             : updated estimate of state mean at time k+1
%	          - PEst             : updated state covariance at time k+1
%             - xPred            : prediction of state mean at time k+1
%             - PPred            : prediction of state covariance at time k+1
%	          - inovation        : innovation vector
%  
% AUTHORS  :  Simon J. Julier       (sjulier@erols.com)    1998-2000
%             Rudolph van der Merwe (rvdmerwe@ece.ogi.edu) 2000
%
% DATE     :  14 August 2000
%
% NOTES    :  The process model is of the form, x(k+1) = ffun[x(k),u(k),v(k),dt]
%             where v(k) is the process noise vector. The observation model is 
%             of the form, z(k) = hfun[x(k),u(k),w(k),dt], where w(k) is the 
%             observation noise vector.
%
%             This code was written to be readable. There is significant
%             scope for optimisation even in Matlab.
%
  

% Process defaults

if (nargin < 10)
  alpha=1;
end;

if (nargin < 11)
  beta=0;
end;

if (nargin < 12)
  kappa=0;
end;

% Calculate the dimensions of the problem and a few useful
% scalars

states       = size(xEst(:),1);
observations = size(z(:),1);
vNoise       = size(Q,2);
wNoise       = size(R,2);


xEstOld = xEst;
PEstOld = PEst;

% Augment the state vector with the noise vectors.
% Note: For simple, additive noise models this part
% can be done differently to save on computational cost.
% For details, contact Rudolph v.d. Merwe

% if (vNoise)
%  PQ=[PEst zeros(states,vNoise);zeros(vNoise,states) Q];
%  xQ=[xEst;zeros(vNoise,1)];
% else
  PQ=PEst;
  xQ=xEst;
% end;

% Calculate the sigma points and there corresponding weights using the Scaled Unscented
% Transformation
[xSigmaPts, wSigmaPts, nsp] = scaledSymmetricSigmaPoints(xQ, PQ, alpha, beta, kappa); 


% Duplicate wSigmaPts into matrix for code speedup
wSigmaPts_xmat = repmat(wSigmaPts(:,2:nsp),states,1);
wSigmaPts_zmat = repmat(wSigmaPts(:,2:nsp),observations,1);


% Work out the projected sigma points and their means
% This routine is fairly generic. The only thing to watch out for are
% angular discontinuities. There is a standard trick for doing this -
% contact me (Julier) for details!

%{
xPredSigmaPts = feval(ffun,xSigmaPts(1:states,:),repmat(U(:),1,nsp), zeros(vNoise, nsp),dt);
% xPredSigmaPts = feval(ffun,xSigmaPts(1:states,:),repmat(U(:),1,nsp), xSigmaPts(states+1:states+vNoise, :),dt);
zPredSigmaPts = feval(hfun,xPredSigmaPts,repmat(U(:),1,nsp), zeros(wNoise, nsp), dt);

% Calculate the mean. Based on discussions with C. Schaefer, form
% is chosen to maximise numerical robustness.
% - I vectorized this part of the code for a speed increase : RvdM 2000

xPred = sum(wSigmaPts_xmat .* (xPredSigmaPts(:,2:nsp) - repmat(xPredSigmaPts(:,1),1,nsp-1)),2);
zPred = sum(wSigmaPts_zmat .* (zPredSigmaPts(:,2:nsp) - repmat(zPredSigmaPts(:,1),1,nsp-1)),2);

xPred=xPred+xPredSigmaPts(:,1);
zPred=zPred+zPredSigmaPts(:,1);

% Work out the covariances and the cross correlations. Note that
% the weight on the 0th point is different from the mean
% calculation due to the scaled unscented algorithm.

exSigmaPt = xPredSigmaPts(:,1)-xPred;
ezSigmaPt = zPredSigmaPts(:,1)-zPred;

PPred   = wSigmaPts(nsp+1)*exSigmaPt*exSigmaPt';
PxzPred = wSigmaPts(nsp+1)*exSigmaPt*ezSigmaPt';
S       = wSigmaPts(nsp+1)*ezSigmaPt*ezSigmaPt';

exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
ezSigmaPt = zPredSigmaPts(:,2:nsp) - repmat(zPred,1,nsp-1);
PPred     = PPred + (wSigmaPts_xmat .* exSigmaPt) * exSigmaPt' + Q;
S         = S + (wSigmaPts_zmat .* ezSigmaPt) * ezSigmaPt' + R;
PxzPred   = PxzPred + exSigmaPt * (wSigmaPts_zmat .* ezSigmaPt)';
%}

%--- Redraw from predicted gaussian
%

xPredSigmaPts = feval(ffun,xSigmaPts,repmat(U(:),1,nsp), zeros(vNoise, nsp),dt);
xPred = sum(wSigmaPts_xmat .* (xPredSigmaPts(:,2:nsp) - repmat(xPredSigmaPts(:,1),1,nsp-1)),2);
xPred = xPred+xPredSigmaPts(:,1);

% exSigmaPt = xPredSigmaPts(:,1)-xPred;
% PPred   = wSigmaPts(nsp+1)*exSigmaPt*exSigmaPt';
% exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
% PPred   = PPred + (wSigmaPts_xmat .* exSigmaPt) * exSigmaPt' + Q;

% Work out the covariances and the cross correlations. Note that
% the weight on the 0th point is different from the mean
% calculation due to the scaled unscented algorithm.

exxSigmaPt = xSigmaPts(:,1)-xEst; % marc
exSigmaPt = xPredSigmaPts(:,1)-xPred;
PxxPred = wSigmaPts(nsp+1)*exxSigmaPt*exSigmaPt'; % marc
PPred   = wSigmaPts(nsp+1)*exSigmaPt*exSigmaPt';

exxSigmaPt = xSigmaPts(:,2:nsp) - repmat(xEst,1,nsp-1); % marc
exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
PPred   = PPred + (wSigmaPts_xmat .* exSigmaPt) * exSigmaPt' + Q;
PxxPred = PxxPred + exxSigmaPt * (wSigmaPts_xmat .* exSigmaPt)'; % marc
crossterm = PxxPred; % marc




% recompute sigma points for measurement mapping
[xPredSigmaPts, wSigmaPts, nsp] = scaledSymmetricSigmaPoints(xPred, PPred, alpha, beta, kappa);
wSigmaPts_zmat = repmat(wSigmaPts(:,2:nsp),observations,1);

zPredSigmaPts = feval(hfun, xPredSigmaPts, repmat(U(:),1,nsp), zeros(wNoise, nsp), dt);
zPred = sum(wSigmaPts_zmat .* (zPredSigmaPts(:,2:nsp) - repmat(zPredSigmaPts(:,1),1,nsp-1)),2);
zPred = zPred+zPredSigmaPts(:,1);

% Work out the covariances and the cross correlations. Note that
% the weight on the 0th point is different from the mean
% calculation due to the scaled unscented algorithm.
exSigmaPt = xPredSigmaPts(:,1)-xPred;
ezSigmaPt = zPredSigmaPts(:,1)-zPred;
PxzPred   = wSigmaPts(nsp+1)*exSigmaPt*ezSigmaPt';
S         = wSigmaPts(nsp+1)*ezSigmaPt*ezSigmaPt';

exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
ezSigmaPt = zPredSigmaPts(:,2:nsp) - repmat(zPred,1,nsp-1);
S         = S + (wSigmaPts_zmat .* ezSigmaPt) * ezSigmaPt' + R;
PxzPred   = PxzPred + exSigmaPt * (wSigmaPts_zmat .* ezSigmaPt)';
%}


%--- Augment sigma points
%{
[xSigmaPts, wSigmaPts, nsp] = scaledSymmetricSigmaPoints(xQ, PQ, alpha, beta, kappa); 
% Duplicate wSigmaPts into matrix for code speedup
wSigmaPts_xmat = repmat(wSigmaPts(:,2:nsp),states,1);

xPredSigmaPts = feval(ffun,xSigmaPts(1:states,:),repmat(U(:),1,nsp), zeros(vNoise, nsp),dt);
xPred = sum(wSigmaPts_xmat .* (xPredSigmaPts(:,2:nsp) - repmat(xPredSigmaPts(:,1),1,nsp-1)),2);
xPred = xPred+xPredSigmaPts(:,1);

exSigmaPt = xPredSigmaPts(:,1)-xPred;
PPred   = wSigmaPts(nsp+1)*exSigmaPt*exSigmaPt';
exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
PPred   = PPred + (wSigmaPts_xmat .* exSigmaPt) * exSigmaPt' + Q;

% exSigmaPt1 = xPredSigmaPts(:,1)-xPred;
% exSigmaPt2 = [xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1),  repmat(xPredSigmaPts(:,1), 1, nsp-1) - repmat(xPred,1,nsp-1)];

[qSigmaPts, tmp, nsp] = scaledSymmetricSigmaPoints(zeros(vNoise,1), Q, alpha, beta, kappa);
xPredSigmaPts = [xPredSigmaPts, repmat(xPredSigmaPts(:,1), 1, nsp-1) + qSigmaPts(:,2:nsp)];
nsp = 2*nsp-1;
[tmp, wSigmaPts, tmp2] = scaledSymmetricSigmaPoints(zeros(2*states,1), [Q, zeros(states); zeros(states), Q], alpha, beta, 3-2*states);
wSigmaPts_zmat = repmat(wSigmaPts(:,2:nsp),observations,1);

zPredSigmaPts = feval(hfun, xPredSigmaPts, repmat(U(:),1,nsp), zeros(wNoise, nsp), dt);
zPred = sum(wSigmaPts_zmat .* (zPredSigmaPts(:,2:nsp) - repmat(zPredSigmaPts(:,1),1,nsp-1)),2);
zPred = zPred+zPredSigmaPts(:,1);

% Work out the covariances and the cross correlations. Note that
% the weight on the 0th point is different from the mean
% calculation due to the scaled unscented algorithm.
exSigmaPt = xPredSigmaPts(:,1)-xPred;
ezSigmaPt = zPredSigmaPts(:,1)-zPred;
S       = wSigmaPts(nsp+1)*ezSigmaPt*ezSigmaPt';
PxzPred = wSigmaPts(nsp+1)*exSigmaPt*ezSigmaPt';
% PxzPred = wSigmaPts(nsp+1)*exSigmaPt1*ezSigmaPt';

exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
ezSigmaPt = zPredSigmaPts(:,2:nsp) - repmat(zPred,1,nsp-1);
S         = S + (wSigmaPts_zmat .* ezSigmaPt) * ezSigmaPt' + R;
PxzPred   = PxzPred + exSigmaPt * (wSigmaPts_zmat .* ezSigmaPt)';
% PxzPred   = PxzPred + exSigmaPt2 * (wSigmaPts_zmat .* ezSigmaPt)';
%}

%%%%% MEASUREMENT UPDATE

% Calculate Kalman gain
K  = PxzPred / S;

% Calculate Innovation
inovation = z - zPred;

% Update mean
xEst = xPred + K*inovation;

% Update covariance
PEst = PPred - K*S*K';

% % for smoothing
J = crossterm / PPred;