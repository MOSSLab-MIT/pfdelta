% solved_cases_path = fullfile('..', 'case14_seeds', 'none');
% mpc_save_path = fullfile(solved_cases_path, 'close2inf_train/generated_mpcs'); 
% raw_hard_save_path = fullfile(solved_cases_path, 'close2inf_train/raw_test'); 
% generate_close2inf(mpc_save_path, raw_hard_save_path);
% 
% 
% 

mpc_last = loadcase("sample_6_lam_0p29392");
mpc_nose = loadcase("sample_6_nose");
fields = fieldnames(mpc_last);

for i = 1:length(fields)
    f = fields{i};
    if ~isequal(mpc_last.(f), mpc_nose.(f))
        fprintf("Field '%s' is different.\n", f);
    end
end

if isequal(mpc_last.gen, mpc_nose.gen)
    disp("gen matrices are identical.")
else
    disp("gen matrices are different.")
    diff = mpc_last.gen - mpc_nose.gen;
    disp("Element-wise difference:")
    disp(diff)
end

% define_constants;
% mpc_b = loadcase("sample_73.m");
% 
% k = 2.5;
% mpc_t = mpc_b;
% mpc_t.gen(:, [PG, QG]) = mpc_b.gen(:, [PG, QG]) * k; 
% mpc_t.bus(:, [PD, QD]) = mpc_b.bus(:, [PD, QD]) * k; 
% mpopt = mpoption('out.all', 0, 'verbose', 2);
% mpopt = mpoption(mpopt, 'cpf.enforce_p_lims', 0, 'cpf.enforce_q_lims', 0, ...
%     'cpf.enforce_v_lims', 0, 'cpf.enforce_flow_lims', 0);  
% mpopt = mpoption(mpopt, 'cpf.stop_at', 'FULL', 'cpf.plot.level', 1, 'cpf.plot.bus', 16, 'cpf.plot.level', 2); 
% mpopt.exp.use_legacy_core = 1;  % <-- force legacy CPF with callback support
% 
% [results, succ] = runcpf(mpc_b, mpc_t, mpopt);
