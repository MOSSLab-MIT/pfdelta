solved_cases_path = fullfile('..', 'my_results', '2025-04-28_11-28-43', 'case14_n');
mpc_save_path = fullfile(solved_cases_path, 'close2inf/generated_mpcs'); 
k = 2.5;
raw_hard_save_path = fullfile(solved_cases_path, 'close2inf/raw'); 

generate_close2inf(mpc_save_path, k, raw_hard_save_path);
