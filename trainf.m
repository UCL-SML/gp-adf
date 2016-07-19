function [hyp, nlml] = trainf(inputs, targets, iter)
%% Code

% 1) Initialization
if nargin < 3, iter = [-500 -1000]; end           % default training iterations

D = size(inputs,2); E = size(targets,2);   % get variable sizes
covfunc = {'covSum', {'covSEard', 'covNoise'}};        % specify ARD covariance
curb.snr = 1000; curb.ls = 100; curb.std = std(inputs);   % set hyp curb

% initialize the hyper-parameters
hyp = zeros(D+2,E); nlml = zeros(1,E);
lh = repmat([log(std(inputs)) 0 -1]',1,E);   % init hyp length scales
lh(D+1,:) = log(std(targets));                      %  signal std dev
lh(D+2,:) = log(std(targets)/10);                     % noise std dev
%

% 2a) Train full GP (always)
fprintf('Train hyper-parameters of full GP ...\n');
for i = 1:E                                          % train each GP separately
  fprintf('GP %i/%i\n', i, E);
  try   % BFGS training
    [hyp(:,i) v] = minimize(lh(:,i), @hypCurb, iter(1), covfunc, ...
      inputs, targets(:,i), curb);
  catch % conjugate gradients (BFGS can be quite aggressive)
    [hyp(:,i) v] = minimize(lh(:,i), @hypCurb, ...
      struct('length', iter(1), 'method', 'CG', 'verbosity', 1), covfunc, ...
      inputs, targets(:,i), curb);
  end
  nlml(i) = v(end);
end
