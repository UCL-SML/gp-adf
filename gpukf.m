function [xEst,PEst,xPred,PPred,zPred,S,K]=...
  gpukf(xEst,PEst,Xd,xd,yd,z,Xm,xm,ym,alpha,beta,kappa)

% TITLE    :  GAUSSIAN PROCESS UNSCENTED KALMAN FILTER
%
% PURPOSE  :  This function performs one complete step of the unscented Kalman filter.
%
% SYNTAX   :  [xEst,PEst,xPred,PPred,zPred,innovation]=ukf(xEst,PEst,U,Q,ffun,z,R,hfun,dt,alpha,beta,kappa)
%
% INPUTS   :  - xEst             : state mean estimate at time k
%             - PEst             : state covariance at time k
%             - U                : vector of control inputs
%             - Xd               : hyper-parameters for GP dynamics model
%             - xd               : training inputs for GP dynamics model
%             - yd               : training targets for GP dynamics model
%             - z                : observation at k+1
%             - Xm               : hyper-parameters for GP observation model
%             - xm               : training inputs for GP observation model
%             - ym               : training targets for GP observation model
%             - alpha (optional) : sigma point scaling parameter. Defaults to 1.
%             - beta  (optional) : higher order error scaling parameter. Default to 0.
%             - kappa (optional) : scalar tuning parameter 1. Defaults to 0.
%
% OUTPUTS  :  - xEst             : updated estimate of state mean at time k+1
%	            - PEst             : updated state covariance at time k+1
%             - xPred            : prediction of state mean at time k+1
%             - PPred            : prediction of state covariance at time k+1
%	            - innovation        : innovation vector
%
% AUTHORS  :  Simon J. Julier       (sjulier@erols.com)    1998-2000
%             Rudolph van der Merwe (rvdmerwe@ece.ogi.edu)      2000
%             Marc Peter Deisenroth                        2008-2009
%
% DATE     :  2009-11-09
%
% NOTES    :  The process model is of the form, x(k+1) = ffun[x(k),u(k),v(k),dt]
%             where v(k) is the process noise vector. The observation model is
%             of the form, z(k) = hfun[x(k),u(k),w(k),dt], where w(k) is the
%             observation noise vector.
%
%             This code was written to be readable. There is significant
%             scope for optimisation even in Matlab.
%

resample = true;

% defaults
covfunc={'covSum',{'covSEard','covNoise'}};


if (nargin < 10)
  alpha=1;
end;

if (nargin < 11)
  beta=0;
end;

if (nargin < 12)
  kappa=0;
end;

Ed = size(yd,2);
Em = size(ym,2);
states = Ed;
observations = size(z(:),1);

% predictive uncertainty
Q = zeros(Ed); % system noise covariance
tmp = zeros(Ed,1);
Xd = reshape(Xd,Ed+2,Ed);
for i = 1:Ed
  [tmp(i) Q(i,i)] = gpr(Xd(:,i),covfunc,xd,yd(:,i),xEst');
end

% measurement uncertainty
R = zeros(Em); % measurement noise covariance
Xm = reshape(Xm,Ed+2,Em);
for i = 1:Em
  [tmp2(i) R(i,i)] = gpr(Xm(:,i),covfunc,xm,ym(:,i),tmp');
end
clear tmp tmp2;



% Calculate the dimensions of the problem and a few useful
% scalars
PQ=PEst;
xQ=xEst;


% Calculate the sigma points and there corresponding weights using the Scaled Unscented
% Transformation
[xSigmaPts, wSigmaPts, nsp] = scaledSymmetricSigmaPoints(xQ, PQ, alpha, beta, kappa);

% Duplicate wSigmaPts into matrix for code speedup
wSigmaPts_xmat = repmat(wSigmaPts(:,2:nsp),states,1);
wSigmaPts_zmat = repmat(wSigmaPts(:,2:nsp),observations,1);

%
% Work out the projected sigma points and their means using the GP models

if ~resample

  for i = 1:Ed
    [xPredSigmaPts(:,i) sxPredSigmaPts] = gpr(Xd(:,i),covfunc,xd,yd(:,i),xSigmaPts(1:states,:)');
  end

  for i = 1:Em
    [zPredSigmaPts(:,i) szPredsigmaPts] = gpr(Xm(:,i),covfunc,xm,ym(:,i),xPredSigmaPts);
  end

  xPredSigmaPts = xPredSigmaPts'; zPredSigmaPts = zPredSigmaPts';

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


  
else
  %--- Redraw from predicted Gaussian
  %
  [xPredSigmaPts sxPredSigmaPts] = gpr(Xd,covfunc,xd,yd,xSigmaPts(1:states,:)');
  xPredSigmaPts = xPredSigmaPts';

  xPred = sum(wSigmaPts_xmat .* (xPredSigmaPts(:,2:nsp) - repmat(xPredSigmaPts(:,1),1,nsp-1)),2);
  xPred = xPred+xPredSigmaPts(:,1);

  exSigmaPt = xPredSigmaPts(:,1)-xPred;
  PPred   = wSigmaPts(nsp+1)*exSigmaPt*exSigmaPt';
  exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
  PPred   = PPred + (wSigmaPts_xmat .* exSigmaPt) * exSigmaPt' + Q;

  [xPredSigmaPts, wSigmaPts, nsp] = scaledSymmetricSigmaPoints(xPred, PPred, alpha, beta, kappa);
  wSigmaPts_zmat = repmat(wSigmaPts(:,2:nsp),observations,1);

  [zPredSigmaPts szPredsigmaPts] = gpr(Xm,covfunc,xm,ym,xPredSigmaPts');
  zPredSigmaPts = zPredSigmaPts';

  zPred = sum(wSigmaPts_zmat .* (zPredSigmaPts(:,2:nsp) - repmat(zPredSigmaPts(:,1),1,nsp-1)),2);
  zPred = zPred+zPredSigmaPts(:,1);

  % Work out the covariances and the cross correlations. Note that
  % the weight on the 0th point is different from the mean
  % calculation due to the scaled unscented algorithm.


  exSigmaPt = xPredSigmaPts(:,1)-xPred;
  ezSigmaPt = zPredSigmaPts(:,1)-zPred;
  PxzPred = wSigmaPts(nsp+1)*exSigmaPt*ezSigmaPt';
  S       = wSigmaPts(nsp+1)*ezSigmaPt*ezSigmaPt';

  exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
  ezSigmaPt = zPredSigmaPts(:,2:nsp) - repmat(zPred,1,nsp-1);
  S         = S + (wSigmaPts_zmat .* ezSigmaPt) * ezSigmaPt' + R;
  PxzPred   = PxzPred + exSigmaPt * (wSigmaPts_zmat .* ezSigmaPt)';
  %}
end

%--- Augment sigma points
%{
%xPredSigmaPts = feval(ffun,xSigmaPts(1:states,:),repmat(U(:),1,nsp), zeros(vNoise, nsp),dt);
[xPredSigmaPts sxPredSigmaPts] = gpr(Xd,covfunc,xd,yd,xSigmaPts(1:states,:)');
xPredSigmaPts = xPredSigmaPts';
xPred = sum(wSigmaPts_xmat .* (xPredSigmaPts(:,2:nsp) - repmat(xPredSigmaPts(:,1),1,nsp-1)),2);
xPred = xPred+xPredSigmaPts(:,1);

exSigmaPt = xPredSigmaPts(:,1)-xPred;
PPred   = wSigmaPts(nsp+1)*exSigmaPt*exSigmaPt';
exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
PPred   = PPred + (wSigmaPts_xmat .* exSigmaPt) * exSigmaPt' + Q;

[qSigmaPts, tmp, nsp] = scaledSymmetricSigmaPoints(zeros(vNoise,1), Q, alpha, beta, kappa);
xPredSigmaPts = [xPredSigmaPts, repmat(xPredSigmaPts(:,1), 1, nsp-1) + qSigmaPts(:,2:nsp)];
nsp = 2*nsp-1;
[tmp, wSigmaPts, tmp2] = scaledSymmetricSigmaPoints(zeros(2*states,1), [Q, zeros(states); zeros(states), Q], alpha, beta, 3-2*states);
wSigmaPts_zmat = repmat(wSigmaPts(:,2:nsp),observations,1);

%zPredSigmaPts = feval(hfun, xPredSigmaPts, repmat(U(:),1,nsp), zeros(wNoise, nsp), dt);
[zPredSigmaPts szPredsigmaPts] = gpr(Xm,covfunc,xm,ym,xPredSigmaPts');
zPredSigmaPts = zPredSigmaPts';
zPred = sum(wSigmaPts_zmat .* (zPredSigmaPts(:,2:nsp) - repmat(zPredSigmaPts(:,1),1,nsp-1)),2);
zPred = zPred+zPredSigmaPts(:,1);

% Work out the covariances and the cross correlations. Note that
% the weight on the 0th point is different from the mean
% calculation due to the scaled unscented algorithm.


exSigmaPt = xPredSigmaPts(:,1)-xPred;
ezSigmaPt = zPredSigmaPts(:,1)-zPred;
PxzPred = wSigmaPts(nsp+1)*exSigmaPt*ezSigmaPt';
S       = wSigmaPts(nsp+1)*ezSigmaPt*ezSigmaPt';

exSigmaPt = xPredSigmaPts(:,2:nsp) - repmat(xPred,1,nsp-1);
ezSigmaPt = zPredSigmaPts(:,2:nsp) - repmat(zPred,1,nsp-1);
S         = S + (wSigmaPts_zmat .* ezSigmaPt) * ezSigmaPt' + R;
PxzPred   = PxzPred + exSigmaPt * (wSigmaPts_zmat .* ezSigmaPt)';
%}


%%%%% MEASUREMENT UPDATE

% Calculate Kalman gain
K  = PxzPred / S;

% Calculate Innovation
innovation = z - zPred;

% Update mean
xEst = xPred + K*innovation;

% Update covariance
PEst = PPred - K*S*K';
