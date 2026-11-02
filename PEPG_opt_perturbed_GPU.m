classdef PEPG_opt_perturbed_GPU < handle
    % GPU-resident PEPG minimizer. Requires Parallel Computing Toolbox.
    % Save this class as PEPG_opt_perturbed_GPU.m.
    % Mean/sigma update equations and defaults match the supplied CPU class.
    % Random samples and arithmetic use GPU single precision.

    properties
        % Core dimensions
        num_params
        matrix_real_dim
        matrix_phase_dim

        % Population settings
        pop_size
        batch_size
        batch_reward

        % Parameters and statistics
        mu
        sigma
        sigma_init

        % Learning rates
        learning_rate
        learning_rate_real
        learning_rate_imaginary
        learning_rate_decay
        learning_rate_limit

        % Sigma adaptation hyperparameters
        sigma_alpha
        sigma_decay
        sigma_limit
        sigma_max_change

        % PEPG options
        forget_best
        rank_fitness
        average_baseline
        use_elite
        elite_ratio

        % State trackers
        first_iteration
        best_reward
        best_mu

        % Perturbations
        epsilon
        epsilon_full
        curr_best_reward
        curr_best_mu
    end

    methods
        function obj = PEPG_opt_perturbed_GPU(num_params, pop_size, learning_rate, learning_rate_real, learning_rate_imaginary, matrix_real_dim, matrix_phase_dim, starting_mu, starting_sigma)
            validateattributes(num_params,{'numeric'},{'scalar','integer','positive'});
            validateattributes(pop_size,{'numeric'},{'scalar','integer','positive'});
            assert(numel(starting_mu)==num_params && numel(starting_sigma)==num_params, ...
                'Starting mu and sigma must each contain num_params elements.');
            assert(matrix_real_dim + matrix_phase_dim == num_params, ...
                'The split learning-rate dimensions must sum to num_params.');
            obj.num_params = num_params;
            obj.matrix_real_dim = matrix_real_dim;
            obj.matrix_phase_dim = matrix_phase_dim;

            obj.pop_size = pop_size + mod(pop_size,2);  % ensure even
            obj.batch_size = obj.pop_size/2;
            obj.batch_reward = gpuArray.zeros(obj.pop_size,1,'single');

            obj.mu = gpuArray(single(starting_mu(:).')); 
            obj.sigma = gpuArray(single(starting_sigma(:).'));
            assert(gather(all(isfinite(obj.sigma) & obj.sigma > 0)), ...
                'Starting sigma must be finite and positive.');
            obj.sigma_init = obj.sigma;

            % Learning rates
            obj.learning_rate = learning_rate;
            obj.learning_rate_real = learning_rate_real;
            obj.learning_rate_imaginary = learning_rate_imaginary;
            obj.learning_rate_decay = 0.99;
            obj.learning_rate_limit = 0.01;

            % Sigma adaptation parameters
            %obj.sigma_alpha = 0.30;
            obj.sigma_alpha = 1;
            %obj.sigma_decay = 0.999;
            obj.sigma_decay = 1;
            obj.sigma_limit = 0.00001;
            obj.sigma_max_change = 0.2;

            % PEPG options
            obj.forget_best = true;
            obj.rank_fitness = true;
            obj.average_baseline = true;
            obj.use_elite = false;
            obj.elite_ratio = 0;

            % Initial state
            obj.first_iteration = true;
            obj.best_reward = gpuArray(single(inf));
            obj.best_mu = obj.mu;
        end

        function ranks = compute_ranks(~, x)
            % Zero-based ranking
            [~, idx] = sort(x);
            ranks = zeros(size(x),'like',x);
            ranks(idx) = cumsum(ones(size(x),'like',x)) - 1;
        end

        function cr = compute_centered_ranks(obj, x)
            y = obj.compute_ranks(x);
            cr = y/(numel(x)-1) - 0.5;
        end

        function [mu, epsilon_full] = ask(obj)
            assert(obj.average_baseline, ...
                'This GPU class supports average_baseline=true (paired populations).');
            % Sample perturbations directly on the selected CUDA device.
            obj.epsilon = gpuArray.randn(obj.batch_size, obj.num_params,'single') .* obj.sigma;
            obj.epsilon_full = [obj.epsilon; -obj.epsilon];
            if obj.average_baseline
                mu = obj.mu;
                epsilon_full = obj.epsilon_full;
            else
                mu = [zeros(1,obj.num_params); obj.mu + obj.epsilon_full];
                epsilon_full = [];
            end
        end

        function tell(obj, reward_table)
            assert(obj.average_baseline,'average_baseline must be true.');
            assert(numel(reward_table)==obj.pop_size,'Reward count must equal pop_size.');
            raw_reward = gpuArray(single(reward_table(:)));
            reward = raw_reward;
            if obj.rank_fitness
                reward = obj.compute_centered_ranks(reward);
            end
            if obj.average_baseline
                b = mean(reward);
            else
                b = reward(1);
                reward = reward(2:end);
            end
            r = reward(1:obj.batch_size) - reward(obj.batch_size+1:end);

            % Track RAW minimized loss, rather than centered ranks (whose
            % minimum is always -0.5). This affects bookkeeping only.
            [current_min,idx] = min(raw_reward);
            best_mu_candidate = obj.mu + obj.epsilon_full(idx,:);
            if obj.first_iteration || obj.forget_best || gather(current_min < obj.best_reward)
                obj.best_reward = current_min;
                obj.best_mu = best_mu_candidate;
            end
            obj.curr_best_reward = current_min;
            obj.curr_best_mu = best_mu_candidate;

            % Update mu with split learning rates
            change_mu = r' * obj.epsilon(1:obj.batch_size,:);
            lr = [single(obj.learning_rate_real)*gpuArray.ones(1,obj.matrix_real_dim,'single'), ...
                  single(obj.learning_rate_imaginary)*gpuArray.ones(1,obj.matrix_phase_dim,'single')];
            obj.mu = obj.mu - lr .* change_mu;

            % Adapt sigma
            if obj.sigma_alpha > 0
                S = (obj.epsilon.^2 - obj.sigma.^2) ./ obj.sigma; % computation of the learning signal
                reward_avg = (reward(1:obj.batch_size) + reward(obj.batch_size+1:end))/2; %  this is the antithetic sampling of the rewards as a mean
                delta_sigma = (reward_avg' * S) / (2 * obj.batch_size); % Relates each reward (MSE) with the signal
                change_sigma = obj.sigma_alpha * delta_sigma;
                change_sigma = max(min(change_sigma, obj.sigma_max_change*obj.sigma), -obj.sigma_max_change*obj.sigma);
                obj.sigma = obj.sigma - change_sigma;
            end

            % Decay sigma and learning rate
            obj.sigma = max(obj.sigma * obj.sigma_decay, obj.sigma_limit);
            if obj.learning_rate > obj.learning_rate_limit
                obj.learning_rate = obj.learning_rate * obj.learning_rate_decay;
            end
            obj.first_iteration = false;
        end
    end
end

