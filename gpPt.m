function [M, S, V] = gpPt(X, input, target, m, s)
%
% Compute joint predictions for multiple GPs with uncertain inputs.
%
% compute exact mean and exact covariance matrix of predictive distribution
%
% X       (column) vector of length E*(D+2)
% input   n by D matrix of inputs
% target  n by E matrix of targets
% m       (column) vector of length D, mean of the test distribution
% s       D by D covariance matrix of the test distribution
% M       (column) vector of length E, mean of the predictive distribution
% S       E by E matrix, covariance of the predictive distribution
% V       D by E covariance between inputs and prediction
%
% compute
% E[p(f(x)|m,s)]
% S[p(f(x)|m,s)]
% cov(x,f(x)|m,s)
%
% includes:
% a) uncertainty about the underlying function (in prediction)
% b) measurement/system noise in the predictive covariance
%
% Copyright (C) 2008-2009 by Marc Peter Deisenroth and Carl Edward Rasmussen,
% 2009-06-25


persistent K iK oldX;
[n, D] = size(input);          % number of examples and dimension of input space
[n, E] = size(target);                % number of examples and number of outputs
X = reshape(X, D+2, E)';

% if necessary: re-compute cashed variables
if numel(X) ~= numel(oldX) || isempty(iK) || sum(any(X ~= oldX)) || numel(iK)~=E*n^2 
  oldX = X;                                               
  iK = zeros(n,n,E); K = iK;
  for i=1:E
    inp = bsxfun(@rdivide,input,exp(X(i,1:D)));
    K(:,:,i) = exp(2*X(i,D+1)-maha(inp,inp)/2);
    L = chol(K(:,:,i)+exp(2*X(i,D+2))*eye(n))';
    iK(:,:,i) = L'\(L\eye(n));
  end
end

k = zeros(n,E); beta = k; M = zeros(E,1); V = zeros(D,E); S = zeros(E);

inp = bsxfun(@minus,input,m');
for i=1:E
  % first some useful intermediate terms
  beta(:,i) = (K(:,:,i)+exp(2*X(i,D+2))*eye(n))\target(:,i);
  R = s+diag(exp(2*X(i,1:D))); t = inp/R;
  l = exp(-sum(t.*inp,2)/2); lb = l.*beta(:,i);
  c = exp(2*X(i,D+1))/sqrt(det(R))*exp(sum(X(i,1:D)));
  M(i) = sum(lb)*c;                                             % predicted mean
  V(:,i) = s*c*t'*lb;                                  % input output covariance
  v = bsxfun(@rdivide,inp,exp(X(i,1:D))); k(:,i) = 2*X(i,D+1)-sum(v.*v,2)/2;
end

% compute covariances
for i=1:E            % first: non-central moments
  ii = bsxfun(@rdivide,inp,exp(2*X(i,1:D)));
  for j=1:i
    R = s*diag(exp(-2*X(i,1:D))+exp(-2*X(j,1:D)))+eye(D); t = 1./sqrt(det(R));
    ij = bsxfun(@rdivide,inp,exp(2*X(j,1:D)));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    A = beta(:,i)*beta(:,j)'; 
    if i==j; A = A - iK(:,:,i); end; A = A.*L;
    S(i,j) = t*sum(sum(A)); S(j,i) = S(i,j);
  end
  % add measurement/system noise
  S(i,i) = S(i,i) + exp(2*X(i,D+1)) + exp(2*X(i,D+2)); 
end
% second: centralize moments
S = S - M*M';                                        
