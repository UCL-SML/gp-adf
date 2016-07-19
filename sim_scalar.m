function [sqmaha nllx nlly rmsex] = sim_scalar(flag1, flag2)

% several filters (EKF, UKF, GP-UKF, GP-ADF) tested on a scalar function
%
% inputs arguments (number of arguments counts, not the value)
% flag1: indicates whether figures shall be drawn
% flag2: indicates whether figures shall be printed
%
% returns:
% sqmaha: square-root of Mahalanobis distance in x-space
% nllx:   Negative log-likelihood in x-space (point-wise)
% nlly:   Negative log-likelihood in y-space (point-wise)
% rmsex:  RMSE in x-space
%
% (C) Marc Deisenroth and Marco Huber, 2009-11-09

% close all; clear functions;
switch nargin
  case 0
    clear all; close all;
    fig = 1;
    printFig = 0;
    randn('seed',2);
    rand('twister',4);
  case 1
    clear all; close all;
    fig = 0;
    printFig = 0;
  case 2
    clear all; close all;
    fig = 1;
    printFig = 1;
    randn('state',2);
    rand('twister',4);
end

% some defaults for the plots
set(0,'defaultaxesfontsize',30);
set(0,'defaultaxesfontunits', 'points')
set(0,'defaulttextfontsize',33);
set(0,'defaulttextfontunits','points')
set(0,'defaultaxeslinewidth',0.1);
set(0,'defaultlinelinewidth',2);
set(0,'DefaultAxesLineStyleOrder','-|--|:|-.');

% Parameters for UKF
alpha = 1; beta  = 0; kappa = 2;

%% Kitagawa-like model
c(1) = 0.5;
c(2) = 25;
c(3) = 5;
c(4) = 0;
c(5) = 2;

% system model
afun2 = @(x,u,n,t) c(1)*x + c(2)*x./(1+x.^2) + c(4)*cos(1.2);
A = @(x,u,n,t) c(1) - 2*c(2)*x.^2./(1 + x.^2)^2 + c(2)./(1 + x.^2);
B = @(x,u,n,t) 1;

% measurement model
hfun2 = @(x,u,n,t) c(3).*sin(c(5)*x);
H = @(x,u,n,t) c(3)*c(5)*cos(c(5)*x);

%%
if nargin == 1 
  % sample some noise variances
  C =  (1*rand(1)+1e-04)^2;  % prior uncertainty
  Cw = (1*rand(1)+1e-04)^2;  % system noise
  Cv = (1*rand(1)+1e-04)^2; % measurement noise
else
  C = 0.5^2;
  Cw = 0.2^2;
  Cv = 0.01^2;
end

afun = @(x,u,n,t) afun2(x,u,n,t) + n;
hfun = @(x,u,n,t) hfun2(x,u,n,t) + n;


%% Learn Models
nd = 100; % size dynamics training set
nm = 100; % size of measurement training set

% covariance function
covfunc={'covSum',{'covSEard','covNoise'}};

% learn dynamics model
xd = 20.*rand(nd,1)-10;
yd = afun(xd, [], 0, []) + sqrt(Cw).*randn(nd,1);
Xd = trainf(xd,yd);  disp(exp(Xd))

if 0
  % plot the dynamics model
  xx = linspace(-10,10,100)';
  [mxx sxx] = gpr(Xd,covfunc,xd,yd,xx);

  figure(1)
  hold on
  f = [mxx+2*sqrt(sxx);flipdim(mxx-2*sqrt(sxx),1)];
  fill([xx; flipdim(xx,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
  plot(xd,yd,'ro','markersize',2);
  plot(xx, mxx, 'k')
  plot(xx,afun(xx, [], 0, []),'b');
  xlabel('input state');
  ylabel('succ. state');
  axis tight
  if printFig; print_fig('dynModel'); end
  disp('this is the learned dynamics model'); 
  disp('press any key to continue');
  pause
end

% learn observation model
xm = 20.*rand(nm,1)-10;
ym = hfun(xm, [], 0, []) + sqrt(Cv).*randn(nm,1);
Xm = trainf(xm,ym);  disp(exp(Xm))

if 0
  % plot the observation model
  xx = linspace(-10,10,100)';
  [mxx sxx] = gpr(Xm,covfunc,xm,ym,xx);

  figure(2); clf
  hold on
  f = [mxx+2*sqrt(sxx);flipdim(mxx-2*sqrt(sxx),1)];
  fill([xx; flipdim(xx,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
  plot(xm,ym,'ro','markersize',2);
  plot(xx, mxx, 'k')
  plot(xx,hfun(xx, [], 0, []),'b');
  xlabel('input state');
  ylabel('observation');
  if printFig; print_fig('measModel'); end
  disp('this is the learned measurement model'); 
  disp('press any key to continue');
  pause
end


%% some error measures
sfun = @(xt, x, C) (x - xt).^2; % squared distance
smfun = @(xt, x, C) (x - xt).^2./C./length(x); % squared Mahalanobis distance per point
mfun = @(xt, x, C) sqrt((x-xt).^2./C)./length(x); % Mahalanobis distance per point
nllfun = @(xt, x, C) (0.5*log(C) + 0.5*(x-xt).^2./C + 0.5.*log(2*pi))./length(x); % NLL per point


%% State estimation
T = 1;        % length of prediction horizon
noTest = 200; % size of test set
x = linspace(-10, 10, noTest); % means of initial states
y = zeros(1, noTest);  % observations

% Considered estimators: (1) ground truth, (2) ukf, (3) gpf, (4) ekf
xp = zeros(5, noTest,T+1); % predicted state
xe = zeros(5, noTest,T+1); % filtered state
xy = zeros(5, noTest,T+1); % predicted measurement (mean)
Cy = zeros(5, noTest,T+1); % predicted measurement (variance)
Cp = zeros(5, noTest,T+1); % predicted variance
Ce = zeros(5, noTest,T+1); % filtered variance

xe(:,:,1) = repmat(x,5,1);
xe(1,:,1) = chol(C)'*randn(noTest,1) + x';
Ce(:,:,1) = repmat(C,5,noTest);
for t = 1:T
  for i = 1:length(x)
    %----------------------------- Ground Truth --------------------------
    w = sqrt(Cw)'*randn(1);
    v = sqrt(Cv)'*randn(1);
    xp(1,i,t+1) = afun(xe(1,i,t), [], w, []);
    xe(1,i,t+1) = xp(1,i,t+1);
    y(i) = hfun(xp(1,i,t+1), [], v, []);
    xy(1,i,t+1) = y(i);
    %--------------------------------- UKF -------------------------------
    [xe(2,i,t+1), Ce(2,i,t+1), xp(2,i,t+1), Cp(2,i,t+1), xy(2,i,t+1), Cy(2,i,t+1)] = ...
      ukf_add(xe(2,i,t), Ce(2,i,t), [], Cw, afun, y(i), Cv, hfun, [], alpha, beta, kappa);
    
    %------------------------------ GP-ADF -------------------------------
    [xe(3,i,t+1), Ce(3,i,t+1), xp(3,i,t+1), Cp(3,i,t+1), xy(3,i,t+1), Cy(3,i,t+1)] = ...
      gpf(Xd, xd, yd, Xm, xm, ym, xe(3,i,t), Ce(3,i,t), y(i));

    %--------------------------------- EKF -------------------------------
    AA = A(xe(4,i,t), [], 0, []); % Jacobian of A
    BB = B(xe(4,i,t), [], 0, []); % Jacobian of B
    xp(4,i,t+1) = afun(xe(4,i,t), [], 0, []);   % predicted state mean
    Cp(4,i,t+1) = AA*Ce(4,i,t)*AA' + BB*Cw*BB'; % predicted state covariance

    HH = H(xp(4,i,t+1), [], 0, []); % Jacobian of H
    xy(4,i,t+1) = hfun(xp(4,i,t+1), [], 0, []); % predicted measurement mean
    Cy(4,i,t+1) = (Cv + HH*Cp(4,i,t+1)*HH'); % predicted measurmeent covariance

    K = Cp(4,i,t+1)*HH'/Cy(4,i,t+1); % Kalman gain
    xe(4,i,t+1) = xp(4,i,t+1) + K*(y(i) - xy(4,i,t+1)); % filter mean
    Ce(4,i,t+1) = Cp(4,i,t+1) - K*HH*Cp(4,i,t+1);       % filter covariance

    %--------------------------------- GP-UKF -------------------------------
    [xe(5,i,t+1), Ce(5,i,t+1), xp(5,i,t+1), Cp(5,i,t+1), xy(5,i,t+1), Cy(5,i,t+1)] = ...
      gpukf(xe(5,i,t), Ce(5,i,t), Xd, xd, yd, y(i), Xm, xm, ym, alpha, beta, kappa);
  end
end

%% Plot

if fig
  figure(3)
  filternames = {'true','UKF', 'GP-ADF', 'EKF','GP-UKF'};
  for i = 2:5
    clf
    hold on

    errorbar(x,xe(i,:,T+1),2*sqrt(Ce(i,:,T+1))','rd');
    plot(x, xe(1, :,T+1), 'ko', 'LineWidth', 2);

    xlabel('\mu_0');
    ylabel('p(x_1|z_1,\mu_0,\sigma^2_0)');

    leg = legend(filternames{i},'true');
    set(leg, 'box', 'off', 'FontSize', 30, 'location', 'southeast');
    axis([-10 10 -30 30])
    if printFig; print_fig(['filterDistr_redraw' filternames{i}]); end
    disp(['this plot shows the filtered state distribution for the ' filternames{i}]);
    disp('press any key to continue');
    pause
  end
end


%% some evaluations
disp('maha (x space)')
disp(['UKF           GP-ADF           EKF           GP-UKF'])
sqmaha(1) = sum(mfun(xe(1,:,T+1), xe(2,:,T+1), Ce(2,:,T+1)));
sqmaha(2) = sum(mfun(xe(1,:,T+1), xe(3,:,T+1), Ce(3,:,T+1)));
sqmaha(3) = sum(mfun(xe(1,:,T+1), xe(4,:,T+1), Ce(4,:,T+1)));
sqmaha(4) = sum(mfun(xe(1,:,T+1), xe(5,:,T+1), Ce(5,:,T+1)));
disp(num2str(sqmaha));


disp('pointwise NLL (x space):')
disp(['UKF           GP-ADF           EKF           GP-UKF'])
nllx(1) = sum(nllfun(xe(1,:,T+1), xe(2,:,T+1), Ce(2,:,T+1)));
nllx(2) = sum(nllfun(xe(1,:,T+1), xe(3,:,T+1), Ce(3,:,T+1)));
nllx(3) = sum(nllfun(xe(1,:,T+1), xe(4,:,T+1), Ce(4,:,T+1)));
nllx(4) = sum(nllfun(xe(1,:,T+1), xe(5,:,T+1), Ce(5,:,T+1)));
disp(num2str(nllx));

disp('RMSE (x space)')
disp(['UKF           GP-ADF           EKF           GP-UKF'])
rmsex(1) = sqrt(mean(sfun(xe(1,:,T+1), xe(2,:,T+1))));
rmsex(2) = sqrt(mean(sfun(xe(1,:,T+1), xe(3,:,T+1))));
rmsex(3) = sqrt(mean(sfun(xe(1,:,T+1), xe(4,:,T+1))));
rmsex(4) = sqrt(mean(sfun(xe(1,:,T+1), xe(5,:,T+1))));
disp(num2str(rmsex));


disp('pointwise NLL (y space):')
disp(['UKF           GP-ADF           EKF           GP-UKF'])
nlly(1) = sum(nllfun(xy(1,:,T+1), xy(2,:,T+1), Cy(2,:,T+1)));
nlly(2) = sum(nllfun(xy(1,:,T+1), xy(3,:,T+1), Cy(3,:,T+1)));
nlly(3) = sum(nllfun(xy(1,:,T+1), xy(4,:,T+1), Cy(4,:,T+1)));
nlly(4) = sum(nllfun(xy(1,:,T+1), xy(5,:,T+1), Cy(5,:,T+1)));
disp(num2str(nlly));



%% print figures
function print_fig(filename)
path = './figures/';
grid on
set(gcf,'PaperSize', [10 6]);
set(gcf,'PaperPosition',[0.1 0.1 10 6]);
print('-depsc2','-r300', [path filename '.eps']);
eps2pdf([path filename '.eps']);
delete([path filename '.eps']);
