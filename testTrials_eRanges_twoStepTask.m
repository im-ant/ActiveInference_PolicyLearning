%% Defining the MDP
% set up and preliminaries
%==========================================================================
clear
rng('default') %Ensure repeatable behaviour

 
% intial states P(s_1)
%--------------------------------------------------------------------------
D{1} = [1 0 0 0 0 0 0]'; % Current location. Start in position 1.
D{2} = [1 1 1 1]';     % Reward location (no idea)

Nf = length(D);
 
for i = 1:Nf
    Ns(i) = length(D{i});
end
 
% Likelihood matrix: P(o_t|s_t)
%I.e. Outcome factor {1}, (outcome row, state{1} row, state{2} row)
%--------------------------------------------------------------------------
prob_correct_cue = .99;
prob_correct_intermediate_cue = .65; %specify initial cue ambiguity here
prob_reward = .9;

%The "where" outcome are identity matrices (agent knows where it is)
A{1}(:,:,1) = eye(Ns(1));
A{1}(:,:,2) = eye(Ns(1));
A{1}(:,:,3) = eye(Ns(1));
A{1}(:,:,4) = eye(Ns(1));

% "information" outcome for different reward locations
for reward_loc = 1:Ns(2)
    %First, initialize everything to zeros
    A{2}(:,:,reward_loc) = zeros(4,7);
end

%Set up cue probability at the initial state
A{2}(1:2,1,:) = 1-prob_correct_intermediate_cue;
A{2}(1,1,1:2) = prob_correct_intermediate_cue;
A{2}(2,1,3:4) = prob_correct_intermediate_cue;

%Set up cue probabilities given at the intermediate states
A{2}(1:2,2:3,:) = 0.0001; %Some small value 
A{2}(1,2,1) = prob_correct_cue;
A{2}(2,2,1) = 1-prob_correct_cue;
A{2}(2,2,2) = prob_correct_cue;
A{2}(1,2,2) = 1-prob_correct_cue;
A{2}(1,3,3) = prob_correct_cue;
A{2}(2,3,3) = 1-prob_correct_cue;
A{2}(2,3,4) = prob_correct_cue;
A{2}(1,3,4) = 1-prob_correct_cue;

%Set up the final reward probability%
%Set everything up with high prob punishment and low prob reward first
A{2}(3,4:7,1:4) = 0.0001;
A{2}(4,4:7,1:4) = 0.9999;
%Set up the accidentally no reward cases
A{2}(3:4,4:5,1:2) = 1-prob_reward;
A{2}(3:4,6:7,3:4) = 1-prob_reward;
%Finally, add only the rewarding cases 
A{2}(3,4,1) = prob_reward;
A{2}(4,5,1) = prob_reward;
A{2}(3,5,2) = prob_reward;
A{2}(4,4,2) = prob_reward;
A{2}(3,6,3) = prob_reward;
A{2}(4,7,3) = prob_reward;
A{2}(3,7,4) = prob_reward;
A{2}(4,6,4) = prob_reward;

 
Ng    = numel(A);
for g = 1:Ng
    No(g)  = size(A{g},1);
end


% Transition matrix: P(s_t+1|s_t,pi)
%I.e. B{state factor number}(finish, start, control number)
%--------------------------------------------------------------------------

%The reward location state is an identity transition.
for f = 1:Nf
    B{f} = eye(Ns(f));
end


%Iterate through the control numbers that facilitates location transition 
for k = 1:Ns(1)
    %First, initialize everything to identity
    B{1}(:,:,k) = eye(Ns(1));
end

%First state can only go to 2 or 3
for k = 2:3
    B{1}(1,1,k) = 0;
    B{1}(k,1,k) = 1;
end

%Intermediate states can only go forward
for k = 4:5
    B{1}(2,2,k) = 0;
    B{1}(k,2,k) = 1;
end
for k = 6:7
    B{1}(3,3,k) = 0;
    B{1}(k,3,k) = 1;
end
%Or back to 1?
%B{1}(1,2:3,1) = 1;

%Final states are absorbing states
B{1}(:,4:7,:) = 0;
for k = 4:7
    B{1}(k,k,:) = 1;
end


 
% Priors preference over outcomes: P(o_t)
%--------------------------------------------------------------------------
c = 2;
C{1} = zeros(No(1),1); %No preference over location
C{2} = [0 0 2*c -c]'; %Prefer reward and dislike punishment



% Policies
% V(j, k, f) = [ selection for each state ], where
%   j = future time points
%   k = plaudible policies
%   f = (going to) state factors (and matrices indicate which B matrix to use) 
%--------------------------------------------------------------------------
%Policies to move around
V(:,:,1) = [2 4 4 4;
            2 5 5 5;
            3 6 6 6;
            3 7 7 7;
            2 2 2 2;
            3 3 3 3;
            1 1 1 1]';
        
%Policies to change reward location - not possible
V(:,:,2)  = ones(size(V(:,:,1)));



% MDP structure
%--------------------------------------------------------------------------
clear mdp
mdp.s = [1; 1];                  % Arbitrary state set-up
mdp.A = A;                      % observation model or likelihood
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states
mdp.T = 5;                      % number of time steps
mdp.V = V;
mdp.alpha  = 10000;

mdp.Aname = {'',''};
mdp.Bname = {'where','context'};

mdp       = spm_MDP_check(mdp);

OPTIONS.gamma = 1;


%% Look at policy choice under different priors (i.e. across days)
test_4_policy_choice = 0; %(1 = true, actually run this section)


if test_4_policy_choice
    rng('shuffle');
    
    %How many times to test each of the prior e parameters?
    N_testTrials = 32;

    %How many training day priors to run through (starts from day 1, -1 for all)
    UpTo_nth_trainingDay = -1;
    
    % Load the matrix with e over time
    % ------------------------------------------
    %e_file = load('matrices/2_level_prune/e_rew45_32DayTraining.mat'); %specialized agent
    %e_file = load('matrices/2_level_prune/e_rew4567_32DayTraining.mat'); %general agent
    %e_file = load('matrices/2_level_prune/e_naive_32DayTraining.mat'); %naive agent
    %e_file = load('matrices/e_256uncertain_singlePolicy_rew3.mat') %uncertain agent
    
    % Other set-ups
    % ------------------------------------------
    % Save accuracy matrix over time (1 = true, save it)
    save_accuracy_mat = 1;
    % The path and filename of the saved output file
    diffPrior_accuracy_path = "accOverDays_genAgent_genEnv.mat";
    
    %Set up an array of reward locaiton states to iterate over
    rewLocs_arr = [1 2 3 4];
    
    
    % Iternal script set-ups
    % ------------------------------------------
    
    %Extract the actual e matrix from the e file
    e_overTime_mat = e_file.e_over_time_arr;
    %Get the total number of days in which the prior is saved
    N_training_days = length(e_overTime_mat);
    
    tic %Time?
    %Figure out how many training days to test for
    n_days = UpTo_nth_trainingDay;
    if ((n_days < 0) || (n_days > N_training_days)) n_days = N_training_days; , end
    
    %Set up matrix to store accuracy after each test set
    proportion_correct_arr = zeros(n_days, 1);
    
    %Iterate through the different days 
    for ith_day = 1:n_days
        ith_day %print-out the number of days tested
        %Initialize the current e prior concentration parameters
        prior_e_concParam = e_overTime_mat(ith_day,:)';
        
        %Counters for correctness
        num_correct = 0;
        
        %Run the number of test trials for this day
        for jth_test = 1:N_testTrials
            %Set up current MDP
            clear cur_MDP
            cur_MDP = mdp;
            %Set up the current e parameter for MDP
            cur_MDP.e = prior_e_concParam;
            
            %Set up the current reward location
            rew_loc = rewLocs_arr( mod(jth_test-1,length(rewLocs_arr))+1 );
            cur_MDP.s = [1; rew_loc];
            
            %Solve current MDP
            cur_MDP = spm_MDP_VB_Xnew(cur_MDP, OPTIONS);
            
            %Check to see if this is a correct trial
            %========================================
            %For conservation, both location and reward outcome needs to happen
            if (cur_MDP.o(2,end)==3)
                num_correct = num_correct + 1;
            end
        end
        
        %Compute proportion correct and record
        %========================================
        p_correct = num_correct / N_testTrials;
        proportion_correct_arr(ith_day) = p_correct;
    end
    
    toc %Show time
    
    % Save file
    % ------------------------------------------
    if save_accuracy_mat
        save(diffPrior_accuracy_path, 'proportion_correct_arr');
    end
    
        
    % Visualize
    % ------------------------------------------
    figure();
    colormap(gray);
    %Visualize the e over time
    subplot(2,1,1);
    plot(e_overTime_mat(1:n_days,:));
    title("prior e parameters");
    xlabel('Days');
    ylabel('e concentration parameter');
    legend();
    %Visualze the correctness over time
    subplot(2,1,2);
    plot(proportion_correct_arr);
    title("Percentage of Correct Trials");
    xlabel('Days');
    ylabel('Proportion of Correct Decisions');
    
end


%% Look a decision making for a single day, generate accuracy distribution
singleDay_accuracy_distribution = 0; %(1 = true for running this section)

if singleDay_accuracy_distribution
    % Load the matrix with e over time
    % ------------------------------------------
    %e_file = load('matrices/e_rew45_32DayTraining.mat'); %specialized agent
    %e_file = load('matrices/e_rew4567_32DayTraining.mat'); %general agent
    %e_file = load('matrices/e_naive_32DayTraining.mat'); %naive agent
    
    % Other set-ups
    % ------------------------------------------
    %How many times to test to get the accuracy percentage score?
    N_testTrials = 32;
    
    %How many times to repeat to generate a distribution? 
    N_reps = 16;

    %Which day's (prior) e parameter to test?
    test_Nth_trainingDay = 33;
    
    % Save accuracy matrix over time (1 = true, save it)
    save_accuracy_mat = 1;
    % The path and filename of the saved output file
    accuracy_distri_path = "accDistr_unamb_naiveAgent_novEnv.mat";
    
    %Set up an array of reward locaiton states to iterate over
    rewLocs_arr = [3 4];
    
    
    % Iternal script set-ups
    % ------------------------------------------
    rng('shuffle');    %Do not have repeatable rng for each trial

    %Extract the actual e matrix from the e file
    e_overTime_mat = e_file.e_over_time_arr;
    
    %Set up matrix to store accuracy after each test set
    proportion_correct_arr = zeros(N_reps, 1);
    
    tic %Time?
    %Iterate through the different repetitions 
    for ith_rep = 1:N_reps
        ith_rep %print-out the number of days tested
        %Initialize the current e prior concentration parameters
        prior_e_concParam = e_overTime_mat(test_Nth_trainingDay,:)';
        
        %Counters for correctness
        num_correct = 0;
        
        %Run the number of test trials for this day
        for jth_test = 1:N_testTrials
            %Set up current MDP
            clear cur_MDP
            cur_MDP = mdp;
            %Set up the current e parameter for MDP
            cur_MDP.e = prior_e_concParam;
            
            %Set up the current reward location
            rew_loc = rewLocs_arr( mod(jth_test-1,length(rewLocs_arr))+1 );
            cur_MDP.s = [1; rew_loc];
            
            %Solve current MDP
            cur_MDP = spm_MDP_VB_Xnew(cur_MDP, OPTIONS);
            
            %Check to see if this is a correct trial
            %========================================
            %Check if a reward was received
            if (cur_MDP.o(2,end)==3)
                num_correct = num_correct + 1;
            end
        end
        
        %Compute proportion correct and record
        %========================================
        p_correct = num_correct / N_testTrials;
        proportion_correct_arr(ith_rep) = p_correct;
    end
    
    toc %Show time
    
    % Save file
    % ------------------------------------------
    if save_accuracy_mat
        save(accuracy_distri_path, 'proportion_correct_arr');
    end
    
    % Visualize
    % ------------------------------------------
    figure();
    boxplot(proportion_correct_arr);
    
    title("Distribution of accuracy");
    ylabel('Accuracy');
    ylim([0 1]);
end

