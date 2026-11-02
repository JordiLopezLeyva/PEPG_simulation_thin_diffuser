% translated from python



classdef PEPG_construct_No_Scattering < handle
    %PEPG_CONSTRUCT_NO_SCATTERING Implements parameterized scattering model
    properties
        N_waveguides
        matrix_real_dim
        matrix_phase_dim
        vector_real_dim
        vector_phase_dim
        x_batch
        y_batch_targ
        pop_size
        scattering_unitary_matrix
        use_complex_boolean
    end
    
    methods
        function obj = PEPG_construct_No_Scattering(N_waveguides, matrix_real_dim, ...
                matrix_phase_dim, vector_real_dim, vector_phase_dim, ...
                x_batch, y_batch_targ, pop_size, scattering_unitary_matrix, use_complex_boolean)
            % Constructor: assign all inputs to properties
            obj.N_waveguides = N_waveguides;
            obj.matrix_real_dim = matrix_real_dim;
            obj.matrix_phase_dim = matrix_phase_dim;
            obj.vector_real_dim = vector_real_dim;
            obj.vector_phase_dim = vector_phase_dim;
            obj.x_batch = x_batch;
            obj.y_batch_targ = y_batch_targ;
            obj.pop_size = pop_size;
            obj.scattering_unitary_matrix = scattering_unitary_matrix;
            obj.use_complex_boolean = use_complex_boolean;
        end
        
        function mats = random_square_matrix(obj, sz, batch_size)
            % Generate batch of random sz-by-sz matrices
            mats = rand(batch_size, sz, sz);
        end
        
        function sel = select_matrix_real_weight(obj, w)
            sel = w(1:obj.matrix_real_dim);
        end
        
        function sel = select_matrix_phase_weight(obj, w)
            idx = obj.matrix_real_dim + (1:obj.matrix_phase_dim);
            sel = w(idx);
        end
        
        function sel = select_input_real_weight(obj, w)
            start = obj.matrix_real_dim + obj.matrix_phase_dim + 1;
            sel = w(start : start + obj.vector_real_dim - 1);
        end
        
        function sel = select_input_phase_weight(obj, w)
            start = obj.matrix_real_dim + obj.matrix_phase_dim + obj.vector_real_dim + 1;
            sel = w(start : start + obj.vector_phase_dim - 1);
        end
        
        function M = triangular_symmetric_matrix(~, vec)
            % Build symmetric matrix from upper-triangle vector
            L = numel(vec);
            n = (-1 + sqrt(1 + 8*L)) / 2;
            if n ~= floor(n) || n*(n+1)/2 ~= L
                error('Vector length is not a valid triangular number.');
            end
            n = floor(n);
            M = zeros(n,n);
            k = 1;
            for i = 1:n
                for j = i:n
                    M(i,j) = vec(k);
                    M(j,i) = vec(k);
                    k = k + 1;
                end
            end
        end
        
        function W = vector_to_matrix(obj, wv)
            % Reshape 1D or 2D array into N_waveguides x N_waveguides (or batch)
            if isvector(wv)
                W = reshape(wv, obj.N_waveguides, obj.N_waveguides);
            elseif ismatrix(wv) && size(wv,2) == obj.N_waveguides^2
                B = size(wv,1);
                W = zeros(B, obj.N_waveguides, obj.N_waveguides);
                for b = 1:B
                    W(b,:,:) = reshape(wv(b,:), obj.N_waveguides, obj.N_waveguides);
                end
            else
                error('Input must be 1D or batch of flattened matrices.');
            end
        end
        
        function v = matrix_to_vector(~, M)
            v = M(:).';
        end
        
        function vec = matrix_complex_to_vector(obj, C)
            real_part = real(C(:));
            imag_part = imag(C(:));
            vec = zeros(obj.matrix_real_dim + obj.matrix_phase_dim + ...
                        obj.vector_real_dim + obj.vector_phase_dim, 1);
            vec(1:obj.matrix_real_dim) = real_part(1:obj.matrix_real_dim);
            if obj.use_complex_boolean
                vec(obj.matrix_real_dim+1 : obj.matrix_real_dim+obj.matrix_phase_dim) = ...
                    imag_part(1:obj.matrix_phase_dim);
            end
        end
        
        function CM = Coupling_matrix_func(obj, coupling_weights, use_complex_boolean)
            % Build coupling matrix from real & (optional) imaginary parts
            mr = obj.select_matrix_real_weight(coupling_weights);
            if use_complex_boolean
                mi = obj.select_matrix_phase_weight(coupling_weights);
                CM = obj.vector_to_matrix(mr + 1j*mi);
            else
                CM = obj.vector_to_matrix(mr);
            end
        end
        
        function CV = Coupling_vector_func(obj, coupling_weights, use_complex_boolean)
            mr = obj.select_matrix_real_weight(coupling_weights);
            if use_complex_boolean
                mi = obj.select_matrix_phase_weight(coupling_weights);
                CV = mr + 1j*mi;
            else
                CV = mr;
            end
        end
        
        function output_vector = Output_func(obj, coupling_weights, batch)
            CM = obj.Coupling_matrix_func(coupling_weights, obj.use_complex_boolean);
            x = obj.x_batch(batch, :).';
            output_vector = CM * x;
        end
        
        function mse_val = MSE(obj, output_vector, batch)
            y = obj.y_batch_targ(batch, :).';
            mse_val = mean(abs(y - output_vector).^2);
        end
        
        function mse_vals = calculate_mse_for_coordinates(obj, coordinates, batch)
            mse_vals = zeros(obj.pop_size,1);
            for i = 1:obj.pop_size
                w = coordinates(:,i);
                out = obj.Output_func(w, batch);
                mse_vals(i) = obj.MSE(out, batch);
            end
        end
        
        function U = generate_complex_symmetric_unitary_matrix(~, sz)
            A = rand(sz) + 1j*rand(sz);
            S = A + A.';  % make symmetric
            [V,D] = eig(S);
            U = V * diag(exp(1j*diag(D))) * V.';
            U = 0.5 * (U + U.');
        end
        
        function flag = is_unitary(~, M, tol)
            if nargin<3, tol = 1e-8; end
            n = size(M,1);
            flag = all( abs(M*M' - eye(n)) < tol, 'all');
        end
        
        function Q = generate_real_unitary_matrix(~, N)
            R = randn(N);
            [Q,Rmat] = qr(R);
            d = diag(Rmat);
            Q = Q * diag(sign(d));
        end
        
        function flag = is_symmetric(~, M)
            flag = isequal(M, M.');
        end
    end
end

