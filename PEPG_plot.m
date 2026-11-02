classdef PEPG_plot < handle
    %PEPG_PLOT Provides plotting utilities for PEPG optimization results

    properties
        average_MSE_batch
        MSE_matrix_1
        PEPG_mean
        mean_real
        mean_imaginary
        sigma_mean
        sigma_std
        sigma_mean_real
        sigma_std_real
        sigma_mean_imaginary
        sigma_std_imaginary
        use_complex_boolean
    end

    methods
        function obj = PEPG_plot(average_MSE_batch, MSE_matrix_1, PEPG_mean, mean_real, mean_imaginary, ...
                                  sigma_mean, sigma_std, sigma_mean_real, sigma_std_real, sigma_mean_imaginary, sigma_std_imaginary, use_complex_boolean)
            % Constructor: copy inputs to properties
            obj.average_MSE_batch      = average_MSE_batch(:);
            obj.MSE_matrix_1           = MSE_matrix_1(:);
            obj.PEPG_mean              = PEPG_mean(:);
            obj.mean_real              = mean_real(:);
            obj.mean_imaginary         = mean_imaginary(:);
            obj.sigma_mean             = sigma_mean(:);
            obj.sigma_std              = sigma_std(:);
            obj.sigma_mean_real        = sigma_mean_real(:);
            obj.sigma_std_real         = sigma_std_real(:);
            obj.sigma_mean_imaginary   = sigma_mean_imaginary(:);
            obj.sigma_std_imaginary    = sigma_std_imaginary(:);
            obj.use_complex_boolean    = use_complex_boolean;
        end

        function plot_average_MSE_batch(obj)
            % Plot best MSE per iteration on log-log scale
            figure;
            loglog(obj.average_MSE_batch);
            xlabel('Iteration', 'FontSize', 24);
            ylabel('Best MSE',    'FontSize', 24);
            if obj.use_complex_boolean
                title('PEPG Optimization on MSE', 'FontSize', 30);
            else
                title('PEPG Optimization on MSE of y (real matrices)', 'FontSize', 16);
            end
            grid on;
            set(gca, 'FontSize', 18);
        end

        function plot_W_MSE_batch1(obj)
            % Plot batch-1 weight MSE per iteration on log-log scale
            figure;
            loglog(obj.MSE_matrix_1);
            xlabel('Iteration', 'FontSize', 18);
            ylabel('Best MSE',  'FontSize', 18);
            if obj.use_complex_boolean
                title('MSE of w for batch 1 (complex matrices)', 'FontSize', 20);
            else
                title('MSE of w for batch 1 (real matrices)',    'FontSize', 20);
            end
            grid on;
            set(gca, 'FontSize', 14);
        end

        function plot_means(obj)
            % Compare PEPG_mean, mean_real, and mean_imaginary trends
            figure('Units','normalized','Position',[0.1 0.1 0.6 0.5]);
            loglog(obj.PEPG_mean,    'DisplayName','PEPG\_mean'); hold on;
            loglog(obj.mean_real,    'DisplayName','mean\_real');
            loglog(obj.mean_imaginary,'DisplayName','mean\_imaginary');
            hold off;
            xlabel('Iteration', 'FontSize', 18);
            ylabel('Value',     'FontSize', 18);
            title('Comparison of PEPG\_mean, mean\_real, and mean\_imaginary', 'FontSize', 20);
            grid on;
            legend('Location','best');
            set(gca, 'FontSize', 14);
        end

        function plot_sigmas(obj)
            % Plot sigma mean and sigma std over iterations
            figure('Units','normalized','Position',[0.1 0.1 0.6 0.5]);
            loglog(obj.sigma_mean, 'DisplayName','mean Sigma'); hold on;
            loglog(obj.sigma_std,  'DisplayName','STD Sigma');
            hold off;
            xlabel('Iteration', 'FontSize', 18);
            ylabel('Sigma Value', 'FontSize', 18);
            title('Mean Sigma and STD Sigma', 'FontSize', 20);
            grid on;
            legend('Location','best');
            set(gca, 'FontSize', 14);
        end

        function plot_sigmas_real_and_imaginary(obj)
            % Plot sigma components for real and imaginary parts
            figure('Units','normalized','Position',[0.1 0.1 0.6 0.5]);
            loglog(obj.sigma_mean_real,      'DisplayName','sigma\_mean\_real');    hold on;
            loglog(obj.sigma_std_real,       'DisplayName','sigma\_std\_real');     
            loglog(obj.sigma_mean_imaginary, 'DisplayName','sigma\_mean\_imaginary');
            loglog(obj.sigma_std_imaginary,  'DisplayName','sigma\_std\_imaginary');
            hold off;
            xlabel('Iteration', 'FontSize', 18);
            ylabel('Sigma Value', 'FontSize', 18);
            title('Sigma (Mean and STD) for Real and Imaginary Parts', 'FontSize', 20);
            grid on;
            legend('Location','best');
            set(gca, 'FontSize', 14);
        end
    end
end
