function generate_close2inf(nose_point_cases_path, k, save_path)
    define_constants;

    % matpower format files in selected nose_point_cases_path
    case_files = dir(fullfile(nose_point_cases_path, '*.m'));

    % Specify cpf options (modify/decide on these later)
    mpopt = mpoption('out.all', 0, 'verbose', 1);
    mpopt = mpoption(mpopt, 'cpf.enforce_p_lims', 0, 'cpf.enforce_q_lims', 0, ...
        'cpf.enforce_v_lims', 0, 'cpf.enforce_flow_lims', 0);  % would v_lims enforcement only at PQ?
    mpopt = mpoption(mpopt, 'cpf.stop_at', 'NOSE', 'cpf.plot.level', 0); 
    mpopt.exp.use_legacy_core = 1;  % <-- force legacy CPF with callback support

    fname = 'cpf_trace';
    solvedcase = 'cpf_solution_nose';

    % Generate nose points.
    for i=1:numel(case_files)

        % Specify current sample net_path
        current_net = case_files(i);
        current_net_path = fullfile(current_net.folder, current_net.name);
        mpc_b = loadcase(current_net_path);
        
        % Specify target for base_mpc
        mpc_t = mpc_b;
        mpc_t.gen(:, [PG, QG]) = mpc_b.gen(:, [PG, QG]) * k; 
        mpc_t.bus(:, [PD, QD]) = mpc_b.bus(:, [PD, QD]) * k; 
        [~, base_name, ~] = fileparts(current_net_path);   
        solvedcase = char(fullfile(save_path, base_name + "_nose.m"));

        % Run CPF
        results = my_runcpf(mpc_b, mpc_t, mpopt, [], solvedcase, current_net_path, save_path);

        if ~(results.cpf.events.name == 'NOSE') % double check that this only has NOSE when nose is reached
            % increase k if NOSE event was not triggered
            delete(fullfile(save_path, [base_name, '_*.m']));
            k = k * 1.5;
            mpc_t = mpc_b;
            mpc_t.gen(:, [PG, QG]) = mpc_b.gen(:, [PG, QG]) * k; 
            mpc_t.bus(:, [PD, QD]) = mpc_b.bus(:, [PD, QD]) * k;
            results = my_runcpf(mpc_b, mpc_t, mpopt, [], solvedcase, current_net_path, save_path); 
        end

        % save k here too. 

    end
end