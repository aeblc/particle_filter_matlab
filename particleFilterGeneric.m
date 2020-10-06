function [particles, weights, particle_mean, particle_covariance] = particleFilterGeneric( ... 
    particles, ...
    weights, ...
    Q, ...
    C, ...
    y, ...
    R, ...
    varargin ...
)
%PARTICLEFILTERGENERIC Generic particle filter implementation
% system equations:
%   x(t) = A * x(t-1) + B * u(t-1) + v
%   y(t) = C * x(t) + w
%
% where;
%  A is the state transition matrix, ( N by N )
%  x(t) is the state at time t, ( N by 1 )
%  B is te input matrix, ( N by U )
%  u(t) is the input vector at time t, ( U by 1 ) 
%  C is the observation matrix, ( M by N )
%  y(t) is the observation at time t, ( M by 1 )
%  v and w are zero mean Gaussian random vectors,
%  
%  Particles are proposals for the current state, they represent the
%  probabilistic nature of the process. (N by K, if there are K particles)
%
%  Particle filter yields an approximation of p(x(t) | x(t-1), u(t-1), y(t)) by sampling 
%  K instances of x(t). Also obtains mean vector and covariance matrix of
%  particles.
%  
%  A, B, u inputs are the last three inputs of this function. Alternatively
%  you can put a function handle instead of A, B and u. Function must return 
%  all updated particles. 



    state_vector_length = numel(particles(:,1));
    particle_count = numel(particles(1,:));
    
    if isa(varargin{1}, 'function_handle')
        
        f = varargin{1};
        
        % get samples from p(x(t) | x(t-1), u(t-1))
        particles = f(particles) + chol(Q) * randn(state_vector_length, particle_count);
        
    elseif isa(varargin{1}, 'numeric')

        A = varargin{1};
        B = varargin{2};
        u = varargin{3};
        
        % get samples from p(x(t) | x(t-1), u(t-1))
        particles = A * particles + B * u + chol(Q) * randn(state_vector_length, particle_count);
        
    end
   
    
    % update weights according to the observation
    % i.e compute p(y(t) | x(t)) * p(x(t) | x(t-1), u(t-1))
    weights = weights .* mvnpdf((C * particles)', y', R');
    
    % normalize weights to yield p(x(t) | x(t-1), u(t-1), y(t))
    weights = weights / sum(weights);
    
    % find effective sample size for optimal sampling
    N_eff = 1 / sum(weights.^2);
    
    % resample if effective sample size is low
    if N_eff < particle_count / 2
        
        weights_cumulative_sum = cumsum(weights);
        indexes = zeros(particle_count, 1);
        
        for i = 1:particle_count
            j = find(weights_cumulative_sum > rand, 1);
            indexes(i) = j;
        end
        
        particles = particles(:, indexes);
        weights = ones(particle_count, 1) / particle_count;
        
    end
    
    % obtain mean and covariances of particles for future use
    particle_mean = particles * weights;
    particle_covariance = cov(particles');
    
end