function bayesianAvg_mR = get_bayesianAvg_reducedPriors(prior_mF,post_mF)
    %Store the number of policies
    N_policies = numel(prior_mF);
    
    %Set up changes to e of the discarded policies (e_discrd = e*[2^(d_exp)])
    disExp_arr = [-2, 0];
    %disExp_arr = [-32];
    
    % Total numbers of reduced models that we will evaluate
    totNum_mR = length(disExp_arr)*(2^N_policies);
    % Initialize array to store the model evidence of each model
    delta_F_arr = zeros(totNum_mR, 1);
    % Initialize array to store the prior mR of each model
    prior_mR_arr = zeros(totNum_mR, N_policies);
    % Counter to count for the reduced model index during iteration
    mR_idx = 1;
    
    %%Go through each level to set the discarded policies to%%
    for disExp = disExp_arr
        %How much to multiply the prior e's by?
        dis_prior_e_factor = 2^disExp;
        
        % Set up the integer for generating all reduced permutations %
        preBin_int = 0;
        num_permu = (2^N_policies);

        %%Go through each permutation and calculate model evidence %%
        while (preBin_int < num_permu)
            %Initialize binary array for which policy to delete
            policy_kept = zeros(N_policies,1);
            %Generate the current policies to keep 
            strBin = dec2bin(preBin_int, N_policies);
            for i = 1:length(strBin)
                policy_kept(i) = str2num(strBin(i));
            end

            %Generate the current reduced model 
            cur_mR = (prior_mF.*dis_prior_e_factor) + (policy_kept.*8);

            %Calculate model evidence
            cur_delta_F = spm_MDP_log_evidence(post_mF, prior_mF, cur_mR);

            %Store the current reduced prior concentrations
            prior_mR_arr(mR_idx,:) = cur_mR;
            %Store the the free energy of the current reduced model
            delta_F_arr(mR_idx) = cur_delta_F;

            %Increment
            preBin_int = preBin_int + 1;
            mR_idx = mR_idx + 1;
        end
    end
    
    %%Computing the P(mR|o)
    % Get the positive model evidence
    model_evidence_arr = delta_F_arr .* (-1);
    % Take the softmax to get P(mR|o)
    prob_mR_arr = spm_softmax(model_evidence_arr);
    
    % OPTIONALLY VISUALIZE THINGS
    if 1==0
        %Visulize model evidence
        figure();
        bar(prob_mR_arr, 'FaceColor',[0.5843 0.8157 0.9882] );
        title("Evidence of reduced models");
        xlabel("Permutation of reduced models");
        ylabel("P( model | outcomes )");
        drawnow;
        %Get best reduced model
        figure();
        [tmpVal, tmpIdx] = max(prob_mR_arr);
        bar( prior_mR_arr(tmpIdx,:), 'FaceColor',[0.5843 0.8157 0.9882]);
        title("Best reduced model");
        xlabel("Policies");
        ylabel("e concentration parameters");
        %Get worse reduced model
        figure();
        [tmpVal, tmpIdx] = min(prob_mR_arr);
        bar( prior_mR_arr(tmpIdx,:), 'FaceColor',[0.5843 0.8157 0.9882]);
        title("Worst reduced model");
        xlabel("Policies");
        ylabel("e concentration parameters");
        %Get random reduced model
        figure();
        bar( prior_mR_arr(98,:), 'FaceColor',[0.5843 0.8157 0.9882]);
        title("Intermediate reduced model");
        xlabel("Policies");
        ylabel("e concentration parameters");
    end
        
    %Compute the bayesian averaged reduced model
    bayesianAvg_mR = sum(prob_mR_arr .* prior_mR_arr)';
    
end