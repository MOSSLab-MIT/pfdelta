% solved_cases_path = fullfile('..', 'case14_seeds', 'none');
% mpc_save_path = fullfile(solved_cases_path, 'close2inf_train/generated_mpcs'); 
% raw_hard_save_path = fullfile(solved_cases_path, 'close2inf_train/raw_test'); 
% generate_close2inf(mpc_save_path, raw_hard_save_path);
% 
% 
% 

mpc = loadcase("sample_9989.m");
result = runpf(mpc)