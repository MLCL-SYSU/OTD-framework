close all
clear
clc

%% 
file = 'coaster2_user02';
path = ['360dataset/sensory/tile/', file, '_tile.csv'];
FoV_data = readmatrix(path, 'range', 'B2:AK1801');
FoV_data(ismissing(FoV_data)) = 0;
T = size(FoV_data, 1);

%% 
E = 100;
M = 10;
N = 20;
M_intile = 3;
N_intile = 3;
R = [1, 2.5, 5, 8, 16, 40] / 40;
K = length(R);

alpha = 1;
beta = 1;

B_mu = M * N * 0.68;
noise_mu = 0;
B_sigma_factor = 0.1;
noise_sigma_factor = 0.1;

Q_factor = 16 / 40;
eta_factor = 1 / 4;
x0_factor = 1;

%% 
A = zeros(M, N, T, E);
A_optimal = zeros(M, N, E);

reg = zeros(T, E);
vio_q = zeros(T, E);
vio_b = zeros(T, E);
vio = zeros(T, E);

%% 
B = min(max(M * N * R(1), B_mu + normrnd(0, B_sigma_factor * B_mu, [T, E])), M * N * R(K));

noise_i = round(noise_mu + normrnd(0, noise_sigma_factor * M * M_intile / 2, [T, E]));
noise_j = round(noise_mu + normrnd(0, noise_sigma_factor * N * N_intile / 2, [T, E]));

%% 
x = linspace(-pi + pi / (N * N_intile), pi - pi / (N * N_intile), N * N_intile);
y = linspace(-pi / 2 + pi / (2 * M * M_intile), pi / 2 - pi / (2 * M * M_intile), M * M_intile)';
theta = cos(y);

%% 
R_vector = vector2extended_vector(R, M, N, K);
DX_h = DX_h_construct(M, N, K);
tda_options_opti = optimoptions("fmincon","MaxFunctionEvaluations",300000,"OptimalityTolerance",1.00e-3);

for e = 1 : E
    %% 
    Q = zeros(1, T);
    FoV_tiles_num = zeros(1, T);
    FoV = zeros(M * M_intile, N * N_intile, T);
    phi = zeros(M * M_intile, N * N_intile, T);
    
    W1_intile = zeros(M * M_intile, N * N_intile, T);
    W2_intile = zeros(M * M_intile, N * N_intile, T);
    W1 = zeros(M, N, T);
    W2 = zeros(M, N, T);
    W3 = zeros(M, N, T);
    W1_vector = zeros(M * N * K, T); 
    W2_vector = zeros(M * N * K, T);   
    W3_vector = zeros(M * N * K, T);
    QoE1 = zeros(1, T);
    QoE2 = zeros(1, T);
    QoE3 = zeros(1, T);
    
    f = zeros(1, T);
    g_q = zeros(1, T);
    g_b = zeros(1, T);
    h = zeros(M * N, T);
    
    X = zeros(M, N, K, T);
    Y = zeros(M, N, K, T);
    X_vector = zeros(M * N * K, T);
    Y_vector = zeros(M * N * K, T);
    lambda_b = zeros(1, T);
    mu = zeros(M * N, T);
    
    X(:, :, x0_factor, 1) = ones(M, N);
    Y(:, :, x0_factor, 1) = ones(M, N);
    Y_vector(:, 1) = tensor2vector(Y(:, :, :, 1), M, N, K);
    
    for t = 1 : T
        fprintf("Experiment %d， Segment %d \n", e, t);

        %%
        X_vector(:, t) = tensor2vector(X(:, :, :, t), M, N, K);
        
        for i = 1 : M
            for j = 1 : N
                index = randsample(K, 1, true, X(i, j, :, t));
                A(i, j, t, e) = R(index) * 40;
            end
        end
        
        %% 
        FoV_data_t = FoV_data(t, :);
        FoV_data_t(FoV_data_t == 0) = [];
        FoV_tiles_num(t) = length(FoV_data_t);
        
        FoV_tile_j =  mod(FoV_data_t, N);
        FoV_tile_j(FoV_tile_j == 0) = N;
        FoV_tile_i = (FoV_data_t - FoV_tile_j) / N + 1;
        
        FoV_tile_i_hat = zeros(1, FoV_tiles_num(t) * M_intile);
        FoV_tile_j_hat = zeros(1, FoV_tiles_num(t) * N_intile);
        for index = 1 : FoV_tiles_num(t)
            for i = 1 : M_intile
                FoV_tile_i_hat(M_intile * (index - 1) + i) = M_intile * (FoV_tile_i(index) - 1) + i + noise_i(t, e);
            end
            for j = 1 : N_intile
                FoV_tile_j_hat(N_intile * (index - 1) + j) = N_intile * (FoV_tile_j(index) - 1) + j + noise_j(t, e);
            end
        end
        
        FoV_tile_i_mod = mod(FoV_tile_i_hat, M * M_intile);
        FoV_tile_i_mod(FoV_tile_i_mod == 0) = M * M_intile;
        FoV_tile_j_mod = mod(FoV_tile_j_hat, N * N_intile);
        FoV_tile_j_mod(FoV_tile_j_mod == 0) = N * N_intile;
        
        FoV(FoV_tile_i_mod, FoV_tile_j_mod, t) = 1;

        %% 
        FoV_C_i = round(mean(FoV_tile_i_hat));
        FoV_C_j = round(mean(FoV_tile_j_hat));
        
        phi(:, :, t) = real(acos(cos(y(FoV_C_i)) * cos(y) * cos(x - x(FoV_C_j)) + sin(y) * sin(y(FoV_C_i))));
        Q(t) = Q_factor * FoV_tiles_num(t);
        
        %% 
        C = sqrt(2 * M * N);  

        Db = R(K) - R(1);
        D = Db;
        
        Gf = (sqrt((1 + alpha^2) * sum(R.^2)) + 2 * beta * R(K)^2) / 3;
        Gb = sqrt(M * N * sum(R.^2)) / sqrt(M * N);
        Gh = sqrt(K) / sqrt(K);
        G = max([Gf, Gb, Gh]);
        
        delta = (M * N + 3) * G^2 + 1;
        eta = sqrt(2) * C / (sqrt(eta_factor * t * ((M * N + 3) * G^2 + 2 * D^2 * (M * N + 2))));
        
        %% 
        w1 = exp(-phi(:, :, t).^2 ./ theta.^2);
        W1_intile(:, :, t) = w1 .* FoV(:, :, t) / sum(sum(w1 .* FoV(:, :, t)));
        W1(:, :, t) = squeeze(sum(sum(reshape(W1_intile(:, :, t), M_intile, M, N_intile, N), 1), 3));
        W1_vector(:, t) = matrix2extended_vector(W1(:, :, t), M, N, K);
        QoE1(t) = (W1_vector(:, t) .* R_vector)' * X_vector(:, t);

        w2 = 1 - exp(-phi(:, :, t).^2 ./ theta.^2);
        W2_intile(:, :, t) = w2 .* (1 - FoV(:, :, t)) / sum(sum(w2 .* (1 - FoV(:, :, t))));
        W2(:, :, t) = squeeze(sum(sum(reshape(W2_intile(:, :, t), M_intile, M, N_intile, N), 1), 3));
        W2_vector(:, t) = matrix2extended_vector(W2(:, :, t), M, N, K);        
        QoE2(t) = -(W2_vector(:, t) .* R_vector)' * X_vector(:, t);
        
        W3_temp = squeeze(sum(sum(reshape(FoV(:, :, t), M_intile, M, N_intile, N), 1), 3));
        W3_temp(W3_temp ~= 0) = 1;
        W3(:, :, t) = W3_temp;
        W3_vector(:, t) = matrix2extended_vector(W3(:, :, t), M, N, K);  
        QoE3(t) = -((W3_vector(:, t) .* R_vector)'.^2  * X_vector(:, t).^2) / (FoV_tiles_num(t));
        
        f(t) = -(QoE1(t) + alpha * QoE2(t) + beta * QoE3(t)) / 3;
        g_q(t) = (Q(t) - (W3_vector(:, t) .* R_vector)' * X_vector(:, t)) / sqrt(FoV_tiles_num(t));
        g_b(t) = (R_vector' * X_vector(:, t) - B(t, e)) / sqrt(M * N);
        h(:, t) = (DX_h' * X_vector(:, t) - 1) / sqrt(K);     

        %%
        DX_f = -(W1_vector(:, t) .* R_vector - alpha * W2_vector(:, t) .* R_vector ...
            - 2 * beta * ((W3_vector(:, t) .* R_vector).^2 .* X_vector(:, t)) / (FoV_tiles_num(t))) / 3;
        DX_g_b = (R_vector) * (g_b(t) > 0) / sqrt(M * N);
        DX_Lagrangian = DX_f + lambda_b(t) * DX_g_b + DX_h * mu(:,t);
        Y_vector(:, t + 1) = X_vector(:, t) - eta * DX_Lagrangian;
        Y(:, :, :, t + 1) = vector2tensor(Y_vector(:, t + 1), M, N, K);
        for i = 1 : M
            for j = 1 : N
                X(i, j, :, t + 1) = projsplx(Y(i, j, :, t + 1));
            end
        end

        Dlambda_b_Lagrangian = max(g_b(t),0) - delta * eta * lambda_b(t);
        lambda_b(t + 1) = max(lambda_b(t) + eta * Dlambda_b_Lagrangian, 0);

        Dmu_Lagrangian = h(t) - delta * eta * mu(:, t);
        mu(:, t + 1) = max(mu(t) + eta * Dmu_Lagrangian, 0);

    end
    
    %% 
    fun = @(x)-(sum(transpose(W1_vector .* R_vector)) * x - alpha * sum(transpose(W2_vector .* R_vector)) * x ...
        - beta * (sum(transpose(W3_vector .* R_vector).^2 ./ transpose(FoV_tiles_num)) * x.^2)) / 3;
    x0 = x0_construct(M, N, K);
    a = [-sum(transpose(W3_vector(:, :) .* R_vector ./ FoV_tiles_num)); T * R_vector' / (M * N)];
    b = [-T * Q_factor; sum(B(:, e)) / (M * N)];
    aeq = DX_h';
    beq = ones(M * N, 1);
    lb = zeros(M * N * K, 1);
    ub = ones(M * N * K, 1);
    X_optimal_vector = fmincon(fun,x0,a,b,aeq,beq,lb,ub,[],tda_options_opti);
    X_optimal(:, :, :) = vector2tensor(X_optimal_vector, M, N, K);

    for i = 1 : M
        for j = 1 : N
            index_optimal = randsample(K, 1, true, X_optimal(i, j, :));
            A_optimal(i, j, e) = R(index_optimal) * 40;
        end
    end  

    %% 
    reg(:, e) = abs(((W1_vector .* R_vector)' * X_optimal_vector - alpha * (W2_vector .* R_vector)' * X_optimal_vector ...
        - beta * (W3_vector .* R_vector)'.^2  ./ FoV_tiles_num' * X_optimal_vector.^2) / 3 + f');
    vio_q(:, e) = max(g_q, 0);
    vio_b(:, e) = max(g_b, 0);
    vio(:, e) = vio_q(:, e) + vio_b(:, e);
end

%% 
A_averE = mean(A, 4);
A_optimal_averE = mean(A_optimal, 3);

reg_averE = mean(reg, 2)';
vio_q_averE = mean(vio_q, 2)';
vio_b_averE = mean(vio_b, 2)';
vio_averE = mean(vio, 2)';

%% 
file_name_format = 'B_OTD_RW_real_%s_T(%d)_E(%d)_noise(mu_%d)_B(mu_%d_sigma_%d)_Q(%d)_';
file_name = [sprintf(file_name_format, file, T, E, noise_mu, 100 * B_mu / (M * N), 10 * B_sigma_factor, 40 * Q_factor), date];
save(file_name);


