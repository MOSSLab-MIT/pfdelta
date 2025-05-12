% solved_cases_path = fullfile('..', 'case14_seeds', 'none');
% mpc_save_path = fullfile(solved_cases_path, 'close2inf_train/generated_mpcs'); 
% raw_hard_save_path = fullfile(solved_cases_path, 'close2inf_train/raw_test'); 
% generate_close2inf(mpc_save_path, raw_hard_save_path);
% 
% 
% 
define_constants;
mpc_b = loadcase("sample_781.m");
k = 2.5;
mpc_t = mpc_b;
mpc_t.gen(:, [PG, QG]) = mpc_b.gen(:, [PG, QG]) * k; 
mpc_t.bus(:, [PD, QD]) = mpc_b.bus(:, [PD, QD]) * k; 
mpopt = mpoption('out.all', 0, 'verbose', 2);
mpopt = mpoption(mpopt, 'cpf.enforce_p_lims', 0, 'cpf.enforce_q_lims', 0, ...
    'cpf.enforce_v_lims', 0, 'cpf.enforce_flow_lims', 0);  
mpopt = mpoption(mpopt, 'cpf.stop_at', 'NOSE', 'cpf.plot.level', 0); 
mpopt.exp.use_legacy_core = 1;  % <-- force legacy CPF with callback support

[results, succ, step_error] = runcpf(mpc_b, mpc_t, mpopt);
