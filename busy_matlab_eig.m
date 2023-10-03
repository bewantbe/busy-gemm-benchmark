% Keep matlab busy, that's it.
% dtype can be 'double' or 'single'
function busy_matlab_eig(n, k_max, dtype, fn_str)
if exist('OCTAVE_VERSION', 'builtin')
  flushstdout = @() fflush(stdout);
  svd_driver('gesdd');  % gesdd, gesvd or gejsv
else
  flushstdout = @() 0;
end
if ~exist('n', 'var') || isempty(n)
  n = 1000;
end
if ~exist('k_max', 'var') || isempty(k_max)
  k_max = 2^48-1;   % upper bound of loop variable in matlab
  k_max = 2^31-3;   % max for Octave 3.8.2
end
if ~exist('dtype', 'var') || isempty(dtype)
  dtype = 'double';
end
if ~exist('fn_str', 'var') || isempty(fn_str)
  fn = @eig;
  fn_str = 'eig';
else
  if strcmp(fn_str, 'eig')
    fn = @eig;
  elseif strcmp(fn_str, 'svd')
    fn = @svd;
  else
    fn = @(X) feval(fn_str, X);
  end
end
fprintf('Computing fn = %s, n = %d, n_loops = %d\n', ...
        fn_str, n, k_max);
randn('state', 1234)   # increase repuducability
gflo = n^3*2;
A = randn(n, n, dtype);   # eig(A) ~ sqrt(n) * unit circle
                          # svd(A) ~ sqrt(n) * [0~2]
radius = norm(A, 'fro');
t00 = tic;
for k = 1 : k_max
  tic;
  lambda = fn(A);
  t = toc;
  s = datestr(now);
  fprintf('%s, t=%.3f, #%d\n', s, t, k);
  %fprintf('mean l = %g\n', mean(abs(lambda)));
  flushstdout();
  % randomized by rank-1 update
  uv = randn(n, 2, dtype) / sqrt(n);
  A = A + 2*rand(1) * uv(:,1) * uv(:,2)';
end
t11 = toc(t00);
fprintf('%s, t=%.3f / %d, ave t=%.3f.\n', s, t, k, t11/k_max);
