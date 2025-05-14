function cpf_success = solve_cpf(current_net_path, save_path)
    define_constants;
    k = 2.5;

    % Specify CPF options
    mpopt = mpoption('out.all', 0, 'verbose', 0);
    mpopt = mpoption(mpopt, 'cpf.enforce_p_lims', 0, 'cpf.enforce_q_lims', 0, ...
        'cpf.enforce_v_lims', 0, 'cpf.enforce_flow_lims', 0);  % would v_lims enforcement only at PQ?
    mpopt = mpoption(mpopt, 'cpf.stop_at', 'NOSE', 'cpf.plot.level', 0); 
    mpopt.exp.use_legacy_core = 1;  % <-- force legacy CPF with callback support

    mpc_b = loadcase(current_net_path);    
    mpc_t = mpc_b;
    mpc_t.gen(:, [PG, QG]) = mpc_b.gen(:, [PG, QG]) * k; 
    mpc_t.bus(:, [PD, QD]) = mpc_b.bus(:, [PD, QD]) * k; 
    [~, base_name, ~] = fileparts(current_net_path);   
    solvedcase = char(fullfile(save_path, base_name + "_nose.m"));
    
    [results, success, step_error] = my_runcpf(mpc_b, mpc_t, mpopt, [], solvedcase, current_net_path, save_path);

    if ~success 
        non_converging_reason = [results.cpf.done_msg, ' for sample ', current_net_path];
    elseif step_error
        non_converging_reason = 'step size too small';
    elseif ~isfield(results.cpf, 'events') || isempty(results.cpf.events)
        non_converging_reason = 'something happened';
    elseif ~(results.cpf.events.name == 'NOSE') 
        non_converging_reason = 'nose event not triggered';
    end

    if ~isfield(results.cpf, 'events')
        cpf_success = false;
    elseif isempty(results.cpf.events)
        cpf_success = false;
    else 
        cpf_success = success && strcmp(results.cpf.events.name, 'NOSE') && ~step_error;
    end

    if exist('non_converging_reason', 'var')
        json_str = jsonencode(non_converging_reason);
        save_path = fullfile(save_path, 'non_converging', [base_name, '_reason.json']);
        fid = fopen(save_path, 'w');
        if fid == -1
            error('Could not open file for writing.');
        end
        fprintf(fid, '%s', json_str);
        fclose(fid);
    end
end