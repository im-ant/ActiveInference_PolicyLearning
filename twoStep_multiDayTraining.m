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
prob_correct_cue = .99; %Secondary cue probability
prob_correct_intermediate_cue = .99; %Initial cue probability
prob_reward = .9; %Final reward probability

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
C{2} = [0 0 3*c -c]'; %Prefer reward and dislike punishment



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


%% Set up a number of identical training days
%Number of days to train
N_days = 64;

%Number of trials in each day
N_trials = 8;

% Drawing (0: draw all figures, 1: draw last day figure, 2: don't draw)
drawAllDays = 1;

%save output matrix or not?
save_e_over_days = 0;
%if so, what to name the output matrix
outputMat_path = 'tmp_e_following_training_32uncertain_shffleRNG.mat';

%Set up initial prior e concentration parameters
mdp.e = ones(length(V),1) .* 1;

%Matrix to keep track of how each policy's e concentration changes over time
e_over_time_arr = zeros(N_days+1, length(V));

tic; %timer starts
%Iterate through each day
for nth_day = 1:N_days
    nth_day
    %Set up the trials for this day
    num_trials = N_trials;
    clear MDP;
    for i = 1:num_trials
        MDP(i) = mdp;
        %Set up true states - regularly alternating between 1 and 2
        rew_loc = (mod(i-1,2)+1);
        %rew_loc = 4; %TODO remove this
        MDP(i).s = [1; rew_loc];
    end
    
    %Record the prior e concentration parameters
    e_over_time_arr(nth_day,:) = mdp.e;
    
    %Solve today's trial
    MDP   = spm_MDP_VB_Xnew(MDP, OPTIONS);
    
    %%Get the bayesian model averaged reduced model (sleep)
    avg_mR_prior = get_bayesianAvg_reducedPriors(mdp.e, MDP(end).e);
    % Generate the posterior model
    post_mR = (MDP(end).e - mdp.e) + avg_mR_prior; 
    
    
    %%VISUALIZE%%
    if (drawAllDays < 2) & ( (drawAllDays == 0) | (nth_day == N_days) )
        %Construct a matrix to hold the belief about (where) states over time
        condExp_states = MDP(1).X{1};
        % Construct a matrix that will hold the policies over time
        condExp_policies = MDP(1).R;
        %Iterate over each MDP to append the expected policy matrices
        for i = 2:num_trials
            %Append state choice
            condExp_states = [condExp_states MDP(i).X{1}];
            %Append policy choice
            condExp_policies = [condExp_policies MDP(i).R];
        end
        %Visualize location states and policy belief
        figure();
        colormap(gray);
        subplot(2,4,[1,2]);
        imagesc(condExp_states);   
        title(['Day ',num2str(nth_day),' Location States']);
        colorbar();
        subplot(2,4,[3,4]);
        imagesc(condExp_policies);
        title("Conditional Expectation of Policies");
        colorbar();

        %Visualize the posterior and the reduced posterior
        ax1 = subplot(2,4,5);
        bar(mdp.e);
        title("Prior e parameter of full model");
        ax2 = subplot(2,4,6);
        bar(MDP(end).e);
        title("Posterior e parameter of full model");
        ax3 = subplot(2,4,7);
        bar(avg_mR_prior);
        title("BMA Reduced Priors");
        ax4 = subplot(2,4,8);
        bar(post_mR);
        title("Reduced Model Posterior e");
        linkaxes([ax1,ax2],'y')
        
        %Optionally draw individual for the daily e
        if 1==0
            %Show the priors well
            figure();
            bar(mdp.e);
            title("Prior e concentration parameters");
            xlabel("Policies");
            ylabel("e concentration parameters");
            ylim([0,18]);
            %Show the posteriors well
            figure();
            bar(MDP(end).e);
            title("Posterior e concentration parameters");
            xlabel("Policies");
            ylabel("e concentration parameters");
            ylim([0,18]);
        end
        %Optionally draw the outcome of bayesian model averaging
        if 1==0
            %Show the BMA prior
            figure();
            bar(avg_mR_prior);
            title("BMA prior model");
            xlabel("Policies");
            ylabel("e concentration parameters");
            ylim([0,18]);
            %Visualize BMA posterior
            figure();
            bar(post_mR);
            title("BMA posterior model");
            xlabel("Policies");
            ylabel("e concentration parameters");
            ylim([0,18]);
        end
        
        
        %Dump visualization now
        drawnow;
    end
    
    %Update the prior concentration parameter for the next run
    mdp.e = post_mR;
end
toc %timer ends

% Get the final e
e_over_time_arr(end,:) = mdp.e;
%Plot how the prior has changed over time
if 1==1
    figure();
    plot(e_over_time_arr);
    title("prior e concentration parameters over time");
    drawnow;
end
%legend();


%Save the e matrix over time and quit
if save_e_over_days
    save(outputMat_path, 'e_over_time_arr');
end

