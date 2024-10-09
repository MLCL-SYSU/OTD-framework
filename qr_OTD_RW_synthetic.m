close all
clear
clc

%% 
T = 40000;
E = 100;
M = 5;
N = 8;
m = 3;
n = 5;
M_intile = 3;
N_intile = 3;
R = [1, 2.5, 5, 8, 16, 40] / 40;
K = length(R);

alpha = 1;
beta = 1;
qr = m * n * 16 / 40;

B_mu = M * N * 0.68;
FoV_C_i_mu = 2.5 * M_intile;
FoV_C_j_mu = 4 * M_intile;
noise_mu = 0;

B_sigma_factor = 0.1;
FoV_C_i_sigma_factor = 0.1;
FoV_C_j_sigma_factor = 0.1;
noise_sigma_factor = 0.1;
eta_factor = 1;
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

FoV_C_i = min(max(1, round(FoV_C_i_mu + normrnd(0, FoV_C_i_sigma_factor * FoV_C_i_mu, [T, 1]))), M * M_intile);
FoV_C_j = min(max(1, round(FoV_C_j_mu + normrnd(0, FoV_C_j_sigma_factor * FoV_C_j_mu, [T, 1]))), N * N_intile);

noise_i = round(noise_mu + normrnd(0, noise_sigma_factor * FoV_C_i_mu, [T, E]));
noise_j = round(noise_mu + normrnd(0, noise_sigma_factor * FoV_C_j_mu, [T, E]));

FoV_C_i_hat = FoV_C_i + noise_i;
FoV_C_j_hat = FoV_C_j + noise_j;

%% 
C = sqrt(2 * M * N);  

Dq = max([abs(Q - m * n * R(1)), abs(Q - m * n * R(K))]) / (m * n);
D = Dq;

Gf = (sqrt((1 + alpha^2) * sum(R.^2)) + 2 * beta * R(K)^2) / 3;
Gq = sqrt(m * n * sum(R.^2)) / sqrt(m * n);
Gh = sqrt(K) / sqrt(K);
G = max([Gf, Gq, Gh]);

delta = (M * N + 3) * G^2 + 1;

%% 
x = linspace(-pi + pi / (N * N_intile), pi - pi / (N * N_intile), N * N_intile);
y = linspace(-pi / 2 + pi / (2 * M * M_intile), pi / 2 - pi / (2 * M * M_intile), M * M_intile)';
theta = cos(y);

phi = zeros(M * M_intile, N * N_intile, T);
for t = 1 : T
    phi(:, :, t) = real(acos(cos(y(FoV_C_i(t))) * cos(y) * cos(x - x(FoV_C_j(t))) + sin(y) * sin(y(FoV_C_i(t)))));
end

%% 
R_vector = vector2extended_vector(R, M, N, K);
DX_h = DX_h_construct(M, N, K);
tda_options_opti = optimoptions("fmincon","MaxFunctionEvaluations",18000,"OptimalityTolerance",1.00e-5);

for e = 1 : E
    %% 
    FoV = zeros(M * M_intile, N * N_intile, T);
    
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
    lambda_q = zeros(1, T);
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
        for i = FoV_C_i_hat(t, e) - (m * M_intile - 1) / 2 : FoV_C_i_hat(t, e) + (m * M_intile - 1) / 2
            i_mod = mod(i, M * M_intile);
            if i_mod == 0
                i_mod = M * M_intile;
            end
            for j = FoV_C_j_hat(t, e) - (n * N_intile - 1) / 2 : FoV_C_j_hat(t, e) + (n * N_intile - 1) / 2
                j_mod = mod(j, N * N_intile);
                if j_mod == 0
                    j_mod = N * N_intile;
                end
                FoV(i_mod, j_mod, t) = 1;
            end
        end
        
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
        QoE3(t) = -((W3_vector(:, t) .* R_vector)'.^2  * X_vector(:, t).^2) / (m * n);
        
        f(t) = -(QoE1(t) + alpha * QoE2(t) + beta * QoE3(t)) / 3;
        g_q(t) = (Q - (W3_vector(:, t) .* R_vector)' * X_vector(:, t)) / sqrt(m * n);
        g_b(t) = (R_vector' * X_vector(:, t) - B(t, e)) / sqrt(M * N);
        h(:, t) = (DX_h' * X_vector(:, t) - 1) / sqrt(K);

        eta = sqrt(2) * C / (sqrt(eta_factor * t * ((M * N + 3) * G^2 + 2 * D^2 * (M * N + 2))));

        %% 
        DX_f = -(W1_vector(:, t) .* R_vector - alpha * W2_vector(:, t) .* R_vector ...
            - 2 * beta * ((W3_vector(:, t) .* R_vector).^2 .* X_vector(:, t)) / (m * n)) / 3;
        DX_g_q = (-W3_vector(:, t) .* R_vector) * (g_q(t) > 0) / sqrt(m * n);
        DX_Lagrangian = DX_f + lambda_q(t) * DX_g_q + DX_h * mu(:,t);
        Y_vector(:, t + 1) = X_vector(:, t) - eta * DX_Lagrangian;
        Y(:, :, :, t + 1) = vector2tensor(Y_vector(:, t + 1), M, N, K);
        for i = 1 : M
            for j = 1 : N
                X(i, j, :, t + 1) = projsplx(Y(i, j, :, t + 1));
            end
        end
        
        Dlambda_q_Lagrangian = max(g_q(t),0) - delta * eta * lambda_q(t);
        lambda_q(t + 1) = max(lambda_q(t) + eta * Dlambda_q_Lagrangian, 0);

        Dmu_Lagrangian = h(t) - delta * eta * mu(:, t);
        mu(:, t + 1) = max(mu(t) + eta * Dmu_Lagrangian, 0);

    end
    
    %% 
    fun = @(x)-(sum(transpose(W1_vector .* R_vector)) * x - alpha * sum(transpose(W2_vector .* R_vector)) * x ...
        - beta * (sum(transpose(W3_vector .* R_vector).^2) * x.^2) / (m * n)) / 3;
    x0 = x0_construct(M, N, K);
    a = [-sum(transpose(W3_vector(:, :) .* R_vector)) / (m * n); T * R_vector' / (M * N)];
    b = [-T * Q / (m * n); sum(B(:, e)) / (M * N)];
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
        - beta * ((W3_vector .* R_vector)'.^2  * X_optimal_vector.^2) / (m * n)) / 3 + f');
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
file_name_format = 'qr_OTD_RW_synthetic_T(%d)_E(%d)_noise(mu_%d)_B(mu_%d_sigma_%d)_Q(%d)_';
file_name = [sprintf(file_name_format, T, E, noise_mu, 100 * B_mu / (M * N), 10 * B_sigma_factor, Q * 40 / (m * n)), date];
save(file_name);


