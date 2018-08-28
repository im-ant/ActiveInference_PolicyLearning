function [MDP] = spm_MDP_VB_Xnew(MDP,OPTIONS)
% active inference and learning using variational message passing
% FORMAT [MDP] = spm_MDP_VB_X(MDP,OPTIONS)
%
% MDP.V(T - 1,P,F)      - P allowable policies (T - 1 moves) over F factors
% or
% MDP.U(1,P,F)          - P allowable actions at each move
% MDP.T                 - number of outcomes
%
% MDP.A{G}(O,N1,...,NF) - likelihood of O outcomes given hidden states
% MDP.B{F}(NF,NF,MF)    - transitions among states under MF control states
% MDP.C{G}(O,T)         - prior preferences over O outsomes in modality G
% MDP.D{F}(NF,1)        - prior probabilities over initial states
% MDP.E(P,1)            - prior probabilities over policies
%
% MDP.a{G}              - concentration parameters for A
% MDP.b{F}              - concentration parameters for B
% MDP.c{G}              - concentration parameters for C
% MDP.d{F}              - concentration parameters for D
% MDP.e                 - concentration parameters for E
%
% optional:
% MDP.s(F,T)            - vector of true states - for each hidden factor
% MDP.o(G,T)            - vector of outcome     - for each outcome modality
% or .O{G}(O,T)         - likelihood matrix of  - for each outcome modality
% MDP.u(F,T - 1)        - vector of actions     - for each hidden factor
%
% MDP.alpha             - precision � action selection [16]
% MDP.beta              - precision over precision (Gamma hyperprior - [1])
% MDP.tau               - time constant for gradient descent [4]
% MDP.eta               - learning rate for a and b parameters
%
% MDP.demi.C            - Mixed model: cell array of true causes (DEM.C)
% MDP.demi.U            - Bayesian model average (DEM.U) see: spm_MDP_DEM
% MDP.link              - link array to generate outcomes from
%                         subordinate MDP; for deep (hierarchical) models
%
% OPTIONS.plot          - switch to suppress graphics:  (default: [0])
% OPTIONS.gamma         - switch to suppress precision: (default: [0])
% OPTIONS.BMR           - Bayesian model reduction for multiple trials
%                         see: spm_MDP_VB_sleep(MDP,BMR)
% produces:
%
% MDP.P(M1,...,MF,T)    - probability of emitting action M1,.. over time
% MDP.Q{F}(NF,T,P)      - expected hidden states under each policy
% MDP.X{F}(NF,T)        - and Bayesian model averages over policies
% MDP.R(P,T)            - conditional expectations over policies
%
% MDP.un          - simulated neuronal encoding of hidden states
% MDP.vn          - simulated neuronal prediction error
% MDP.xn          - simulated neuronal encoding of policies
% MDP.wn          - simulated neuronal encoding of precision (tonic)
% MDP.dn          - simulated dopamine responses (phasic)
% MDP.rt          - simulated reaction times
%
% MDP.F           - (Np x T) (negative) free energies over time
% MDP.G           - (Np x T) (negative) expected free energies over time
%
% This routine provides solutions of active inference (minimisation of
% variational free energy) using a generative model based upon a Markov
% decision process. The model and inference scheme is formulated
% in discrete space and time. This means that the generative model (and
% process) are  finite state machines or hidden Markov models whose
% dynamics are given by transition probabilities among states and the
% likelihood corresponds to a particular outcome conditioned upon
% hidden states.

% When supplied with outcomes (in terms of their likelihood (O) in the
% absence of any policy specification, this scheme will use variational
% message passing to optimise expectations about latent or hidden states
% (and likelihood (A) and prior (B) probabilities. In other words, it will
% invert a hidden Markov model. When  called with policies it will generate
% outcomes that are used to infer optimal policies for active inference.
%
% This implementation equips agents with the prior beliefs that they will
% maximise expected free energy: expected free energy is the free energy
% of future outcomes under the posterior predictive distribution. This can
% be interpreted in several ways � most intuitively as minimising the KL
% divergence between predicted and preferred outcomes (specified as prior
% beliefs) � while simultaneously minimising ambiguity.
%
% This particular scheme is designed for any allowable policies or control
% sequences specified in MDP.V. Constraints on allowable policies can limit
% the numerics or combinatorics considerably. Further, the outcome space
% and hidden states can be defined in terms of factors; corresponding to
% sensory modalities and (functionally) segregated representations,
% respectively. This means, for each factor or subset of hidden states
% there are corresponding control states that determine the transition
% probabilities.
%
% This specification simplifies the generative model, allowing a fairly
% exhaustive model of potential outcomes. In brief, the agent encodes
% beliefs about hidden states in the past (and in the future) conditioned
% on each policy. The conditional expectations determine the (path
% integral) of free energy that then determines the prior over policies.
% This prior is used to create a predictive distribution over outcomes,
% which specifies the next action.
%
% In addition to state estimation and policy selection, the scheme also
% updates model parameters; including the state transition matrices,
% mapping to outcomes and the initial state. This is useful for learning
% the context. Likelihood and prior probabilities can be specified in terms
% of concentration parameters (of a Dirichlet distribution (a,b,c,..). If
% the corresponding (A,B,C,..) are supplied  they will be used to generate
% outcomes; unless called without policies (in hidden Markov model mode).
% In this case, the (A,B,C,..) are treated as posterior estimates.
%
% See also: spm_MDP, which uses multiple future states and a mean field
% approximation for control states � but allows for different actions
% at all times (as in control problems).
%
% See also: spm_MDP_game_KL, which uses a very similar formulation but just
% maximises the KL divergence between the posterior predictive distribution
% over hidden states and those specified by preferences or prior beliefs.
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_VB_X.m 7294 2018-04-16 15:48:57Z thomas $


% deal with a sequence of trials
%==========================================================================

% options
%--------------------------------------------------------------------------
try, OPTIONS.plot;  catch, OPTIONS.plot  = 0; end
try, OPTIONS.gamma; catch, OPTIONS.gamma = 0; end

% if there are multiple trials ensure that parameters are updated
%--------------------------------------------------------------------------
if length(MDP) > 1
    
    OPTS      = OPTIONS;
    OPTS.plot = 0;
    for i = 1:length(MDP)
        
        % update concentration parameters
        %------------------------------------------------------------------
        if i > 1
            try,  MDP(i).a = OUT(i - 1).a; end
            try,  MDP(i).b = OUT(i - 1).b; end
            try,  MDP(i).d = OUT(i - 1).d; end
            try,  MDP(i).c = OUT(i - 1).c; end
            try,  MDP(i).e = OUT(i - 1).e; end
        end
        
        % solve this trial
        %------------------------------------------------------------------
        OUT(i) = spm_MDP_VB_Xnew(MDP(i),OPTS);
        
        % Bayesian model reduction
        %------------------------------------------------------------------
        if isfield(OPTIONS,'BMR')
            OUT(i) = spm_MDP_VB_sleep(OUT(i),OPTIONS.BMR);
        end
        
    end
    MDP = OUT;
    
    % plot summary statistics - over trials
    %----------------------------------------------------------------------
    if OPTIONS.plot
        if ishandle(OPTIONS.plot)
            figure(OPTIONS.plot); clf
        else
            spm_figure('GetWin','MDP'); clf
        end
        spm_MDP_VB_game(MDP)
    end
    return
end


% set up and preliminaries
%==========================================================================
if isfield(MDP,'U')
    
    % called with repeatable actions (U,T)
    %----------------------------------------------------------------------
    T   = MDP.T;                      % number of updates
    V   = MDP.U;                      % allowable actions (1,Np)
    HMM = 0;
    
elseif isfield(MDP,'V')
    
    % full sequential policies (V)
    %----------------------------------------------------------------------
    V   = MDP.V;                      % allowable policies (T - 1,Np)
    T   = size(MDP.V,1) + 1;          % number of transitions
    HMM = 0;
    
elseif isfield(MDP,'O')
    
    % no policies � assume hidden Markov model (HMM)
    %----------------------------------------------------------------------
    T   = size(MDP.O{1},2);           % HMM mode
    V   = ones(T - 1,1);              % single 'policy'
    HMM = 1;
else
    sprintf('Please specify MDP.U,MDP.V or MDP.O'), return
end

% ensure ppolicy length is less than the number of updates
%--------------------------------------------------------------------------
if size(V,1) > (T - 1)
    V = V(1:(T - 1),:,:);
end

% fill in (posterior or  process) likelihood and priors
%--------------------------------------------------------------------------
if ~isfield(MDP,'A'), MDP.A = MDP.a; end
if ~isfield(MDP,'B'), MDP.B = MDP.b; end

% check format of likelihood and priors
%--------------------------------------------------------------------------
if ~iscell(MDP.A), MDP.A = {full(MDP.A)}; end
if ~iscell(MDP.B), MDP.B = {full(MDP.B)}; end

if isfield(MDP,'a'), if ~iscell(MDP.a), MDP.a = {full(MDP.a)}; end; end
if isfield(MDP,'b'), if ~iscell(MDP.b), MDP.b = {full(MDP.b)}; end; end

% numbers of transitions, policies and states
%--------------------------------------------------------------------------
Ng  = numel(MDP.A);                 % number of outcome factors
Nf  = numel(MDP.B);                 % number of hidden state factors
Np  = size(V,2);                    % number of allowable policies
for f = 1:Nf
    Ns(f) = size(MDP.B{f},1);       % number of hidden states
    Nu(f) = size(MDP.B{f},3);       % number of hidden controls
end
for g = 1:Ng
    No(g) = size(MDP.A{g},1);       % number of outcomes
end

% parameters of generative model and policies
%==========================================================================

% likelihood model (for a partially observed MDP)
%--------------------------------------------------------------------------
p0    = exp(-16);
for g = 1:Ng
    
    % ensure probabilities are normalised : A
    %----------------------------------------------------------------------
    MDP.A{g}  = spm_norm(MDP.A{g});
    
    % parameters (concentration parameters): a
    %----------------------------------------------------------------------
    if isfield(MDP,'a')
        A{g}  = spm_norm(MDP.a{g});
        rA{g} = spm_back(MDP.a{g});
    else
        A{g}  = spm_norm(MDP.A{g});
        rA{g} = spm_back(MDP.A{g});
    end
    
    % (polygamma) function for complexity (and novelty)
    %------------------------------------------------------------------
    if isfield(MDP,'a')
        qA{g} = spm_psi(MDP.a{g} + 1/16);
        pA{g} = MDP.a{g};
        wA{g} = 1./spm_cum(MDP.a{g}) - 1./(MDP.a{g} + p0);
        wA{g} = wA{g}.*(MDP.a{g} > 0);
    end
    
end

% transition probabilities (priors)
%--------------------------------------------------------------------------
for f = 1:Nf
    for j = 1:Nu(f)
        
        % controlable transition probabilities : B
        %------------------------------------------------------------------
        MDP.B{f}(:,:,j) = spm_norm(MDP.B{f}(:,:,j));
        
        % parameters (concentration parameters): b
        %------------------------------------------------------------------
        if isfield(MDP,'b') && ~HMM
            sB{f}(:,:,j) = spm_norm(MDP.b{f}(:,:,j)  + p0);
            rB{f}(:,:,j) = spm_norm(MDP.b{f}(:,:,j)' + p0);
        else
            sB{f}(:,:,j) = spm_norm(MDP.B{f}(:,:,j)  + p0);
            rB{f}(:,:,j) = spm_norm(MDP.B{f}(:,:,j)' + p0);
        end

    end
    
    % (polygamma) function for complexity
    %------------------------------------------------------------------
    if isfield(MDP,'b')
        qB{f} = spm_psi(MDP.b{f} + 1/16);
        pB{f} = MDP.b{f};
    end
    
end


% priors over initial hidden states - concentration parameters
%--------------------------------------------------------------------------
for f = 1:Nf
    if isfield(MDP,'d')
        D{f} = spm_norm(MDP.d{f});
    elseif isfield(MDP,'D')
        D{f} = spm_norm(MDP.D{f});
    else
        D{f} = spm_norm(ones(Ns(f),1));
        MDP.D{f} = D{f};
    end
    
    % (polygamma) function for complexity
    %------------------------------------------------------------------
    if isfield(MDP,'d')
        qD{f} = spm_psi(MDP.d{f} + 1/16);
        pD{f} = MDP.d{f};
    end
end

% priors over policies - concentration parameters
%--------------------------------------------------------------------------
if isfield(MDP,'e')
    E = spm_norm(MDP.e);
elseif isfield(MDP,'E')
    E = spm_norm(MDP.E);
else
    E = spm_norm(ones(Np,1));
end

% (polygamma) function for complexity
%------------------------------------------------------------------
if isfield(MDP,'e')
    qE = spm_psi(MDP.e + 1/16);
    pE = MDP.e;
else
    qE    = spm_log(E);
end



% prior preferences (log probabilities) : C
%--------------------------------------------------------------------------
for g = 1:Ng
    if isfield(MDP,'c')
        Vo{g} = spm_psi(MDP.c{g} + 1/16);
        pC{g} = MDP.c{g};
    elseif isfield(MDP,'C')
        Vo{g} = MDP.C{g};
    else
        Vo{g} = zeros(No(g),1);
    end
    
    % assume time-invariant preferences, if unspecified
    %----------------------------------------------------------------------
    if size(Vo{g},2) == 1
        Vo{g} = repmat(Vo{g},1,T);
    end
    Vo{g}     = spm_log(spm_softmax(Vo{g}));
end

% precision defaults
%--------------------------------------------------------------------------
try, alpha = MDP.alpha; catch, alpha = 16;   end
try, beta  = MDP.beta;  catch, beta  = 1;    end
try, eta   = MDP.eta;   catch, eta   = 1;    end
try, tau   = MDP.tau;   catch, tau   = 4;    end
try, chi   = MDP.chi;   catch, chi   = 1/64; end

% initialise  posterior expectations of hidden states
%--------------------------------------------------------------------------
Ni    = 16;                         % number of VB iterations
for f = 1:Nf
    xn{f} = zeros(Ni,Ns(f),1,1,Np) + 1/Ns(f);
    vn{f} = zeros(Ni,Ns(f),1,1,Np);
    x{f}  = zeros(Ns(f),T,Np)      + 1/Ns(f);
    X{f}  = repmat(D{f},1,1);
    for k = 1:Np
        x{f}(:,1,k) = D{f};
    end
end

% initialise posteriors over polices and action
%--------------------------------------------------------------------------
P  = zeros([Nu,1]);
un = zeros(Np,1);
u  = zeros(Np,1);
a  = zeros(Nf,1);

% If there is only one policy
%--------------------------------------------------------------------------
if Np == 1
    u = ones(Np,T);
end

% expected rate parameter
%--------------------------------------------------------------------------
p     = 1:Np;                       % allowable policies
qbeta = beta;                       % initialise rate parameters
gu    = 1/qbeta;                    % posterior precision (policy)

% solve
%==========================================================================
for t = 1:T
    
    % generate hidden states and outcomes, if in active inference mode
    %======================================================================
    if ~HMM
        
        % sampled state - based on previous action
        %------------------------------------------------------------------
        for f = 1:Nf
            try
                s(f,t) = MDP.s(f,t);
            catch
                if t > 1
                    ps = MDP.B{f}(:,s(f,t - 1),a(f,t - 1));
                else
                    ps = spm_norm(MDP.D{f});
                end
                s(f,t) = find(rand < cumsum(ps),1);
            end
        end
        
        % sample outcome from hidden state (ind), if not specified
        %------------------------------------------------------------------
        ind   = num2cell(s(:,t));
        for g = 1:Ng
            try
                o(g,t) = MDP.o(g,t);
            catch
                po     = MDP.A{g}(:,ind{:});
                o(g,t) = find(rand < cumsum(po),1);
            end
        end
        
        % posterior predictive density (prior for suborinate level)
        %------------------------------------------------------------------
        if isfield(MDP,'link') || isfield(MDP,'demi')
            for f = 1:Nf
                if t > 1
                    xq{f} = sB{f}(:,:,a(f,t - 1))*X{f}(:,t - 1);
                else
                    xq{f} = X{f}(:,t);
                end
            end
        end
        
    else
        
        %  hidden states, outcomes and actions are not needed
        %------------------------------------------------------------------
        s = [];
        o = [];
        a = [];
        
    end
    
    % get outcome likelihood (O)
    %----------------------------------------------------------------------
    for g = 1:Ng
        
        % from posterior predictive density
        %------------------------------------------------------------------
        if isfield(MDP,'link') || isfield(MDP,'demi')
            O{g,t} = spm_dot(A{g},xq);
            
            % specified as a likelihood or observation
            %------------------------------------------------------------------
        elseif isfield(MDP,'O')
            O{g,t} = MDP.O{g}(:,t);
        else
            O{g,t} = sparse(o(g,t),1,1,No(g),1);
        end
    end
    
    % generate outcomes from a subordinate MDP
    %======================================================================
    if isfield(MDP,'link')
        
        % use previous inversions (if available) to reproduce outcomes
        %------------------------------------------------------------------
        try
            mdp = MDP.mdp(t);
        catch
            mdp = MDP.MDP;
        end
        link       = MDP.link;
        mdp.factor = find(any(link,2));
        
        % priors over states (of subordinate level)
        %------------------------------------------------------------------
        for f = 1:size(link,1)
            i = find(link(f,:));
            if numel(i)
                
                % empirical priors
                %----------------------------------------------------------
                mdp.D{f} = O{i,t};
                
                % hidden state for lower level is the outcome
                %----------------------------------------------------------
                try
                    mdp.s(f,1) = mdp.s(f,1);
                catch
                    mdp.s(f,1) = o(i,t);
                end
                
            else
                
                % otherwise use subordinate priors over states
                %----------------------------------------------------------
                try
                    mdp.s(f,1) = mdp.s(f,1);
                catch
                    if isfield(mdp,'D')
                        ps = spm_norm(mdp.D{f});
                    else
                        ps = spm_norm(ones(Ns(f),1));
                    end
                    mdp.s(f,1) = find(rand < cumsum(ps),1);
                end
            end
        end
        
        % infer hidden states at the lower level (outcomes at this level)
        %==================================================================
        MDP.mdp(t) = spm_MDP_VB_Xnew(mdp);
        
        % get inferred outcomes from subordinate MDP
        %------------------------------------------------------------------
        for g = 1:Ng
            i = find(link(:,g));
            if numel(i)
                O{g,t} = MDP.mdp(t).X{i}(:,1);
            end
        end
    end
    
    % generate outcomes from a generalised Bayesian filter
    %======================================================================
    if isfield(MDP,'demi')
        
        % use previous inversions (if available)
        %------------------------------------------------------------------
        try
            MDP.dem(t) = spm_ADEM_update(MDP.dem(t - 1));
        catch
            MDP.dem(t) = MDP.DEM;
        end
        
        % get inferred outcome (from Bayesian filtering)
        %------------------------------------------------------------------
        MDP.dem(t) = spm_MDP_DEM(MDP.dem(t),MDP.demi,O(:,t),o(:,t));
        for g = 1:Ng
            O{g,t} = MDP.dem(t).X{g}(:,end);
        end
    end
    
    
    % Variational updates
    %======================================================================
    if ~HMM || T == t                   % skip accumulation if in HMM mode
        
        % processing time and reset
        %------------------------------------------------------------------
        tstart = tic;
        for f = 1:Nf
            x{f} = spm_softmax(spm_log(x{f})/4);
        end
        
        % Variational updates (hidden states) under sequential policies
        %==================================================================
        S     = size(V,1) + 1;
        F     = zeros(Np,1);
        for k = p                       % loop over plausible policies
            dF    = 1;                  % reset criterion for this policy
            for i = 1:Ni                % iterate belief updates
                F(k)  = 0;              % reset free energy for this policy
                for j = 1:S             % loop over future time points
                    
                    % marginal likelihood over outcome factors
                    %------------------------------------------------------
                    if j <= t
                        for f = 1:Nf
                            xq{f} = x{f}(:,j,k);
                        end
                        for g = 1:Ng
                            Ao{g} = spm_dot(A{g},[O(g,j) xq],(1:Nf) + 1);
                        end
                    end
                    
                    for f = 1:Nf
                        
                        % hidden states for this time and policy
                        %--------------------------------------------------
                        sx = x{f}(:,j,k);
                        v  = spm_zeros(sx);
                        
                        % evaluate free energy and gradients (v = dFdx)
                        %--------------------------------------------------
                        if dF > 0
                            
                            % marginal likelihood over outcome factors
                            %----------------------------------------------
                            if j <= t
                                for g = 1:Ng
                                    Aq = spm_dot(Ao{g},xq,f);
                                    v  = v + spm_log(Aq(:));
                                end
                            end
                            
                            % entropy
                            %----------------------------------------------
                            qx  = spm_log(sx);
                            
                            % emprical priors
                            %----------------------------------------------
                            if j < 2, v = v - qx + spm_log(D{f});                                    end
                            if j > 1, v = v - qx + spm_log(sB{f}(:,:,V(j - 1,k,f))*x{f}(:,j - 1,k)); end
                            if j < S, v = v - qx + spm_log(rB{f}(:,:,V(j    ,k,f))*x{f}(:,j + 1,k)); end
                        
                            % (negative) expected free energy
                            %----------------------------------------------
                            F(k) = F(k) + sx'*v/Nf;
                            
                            % update
                            %----------------------------------------------
                            sx   = spm_softmax(qx + v/tau);
                            
                        else
                            F(k) = G(k);
                        end
                        
                        % store update neuronal activity
                        %--------------------------------------------------
                        x{f}(:,j,k)      = sx;
                        xn{f}(i,:,j,t,k) = sx;
                        vn{f}(i,:,j,t,k) = v - mean(v);
                        
                    end
                end
                
                % convergence
                %----------------------------------------------------------
                if i > 1
                    dF = F(k) - G(k);
                end
                G = F;
                
            end
        end
        
        % accumulate expected free energy of policies (Q)
        %==================================================================
        Q     = zeros(Np,1);
        if Np > 1
            for k = p
                for j = 1:S
                    
                    % get expected states for this policy and time point
                    %------------------------------------------------------
                    for f = 1:Nf
                        xq{f} = x{f}(:,j,k);
                    end
                    
                    % (negative) expected free energy
                    %======================================================
                    
                    % Bayesian surprise about states
                    %------------------------------------------------------
                    Q(k) = Q(k) + spm_MDP_G(A,xq);
                    
                    for g = 1:Ng
                        
                        % prior preferences about outcomes
                        %--------------------------------------------------
                        qo   = spm_dot(A{g},xq);
                        Q(k) = Q(k) + qo'*(Vo{g}(:,j));
                        
                        % Bayesian surprise about parameters
                        %--------------------------------------------------
                        if isfield(MDP,'a')
                            Q(k) = Q(k) - spm_dot(wA{g},[qo xq]);
                        end
                    end
                end
            end
            
            
            % eliminate unlikely policies
            %--------------------------------------------------------------
            if ~isfield(MDP,'U')
                p = p((F(p) - max(F(p))) > -3);
            else
                OPTIONS.gamma = 1;
            end
            
            % variational updates - policies and precision
            %==============================================================
            
            % previous expected precision
            %--------------------------------------------------------------
            if t > 1
                gu(t) = gu(t - 1);
            end
            for i = 1:Ni
                
                % posterior and prior beliefs about policies
                %----------------------------------------------------------
                qu = spm_softmax(qE(p) + gu(t)*Q(p) + F(p));
                pu = spm_softmax(qE(p) + gu(t)*Q(p));
                
                % precision (gu) with free energy gradients (v = -dF/dw)
                %----------------------------------------------------------
                if OPTIONS.gamma
                    gu(t) = 1/beta;
                else
                    eg    = (qu - pu)'*Q(p);
                    dFdg  = qbeta - beta + eg;
                    qbeta = qbeta - dFdg/2;
                    gu(t) = 1/qbeta;
                end
                
                % simulated dopamine responses (precision at each iteration)
                %----------------------------------------------------------
                n       = (t - 1)*Ni + i;
                wn(n,1) = gu(t);
                un(p,n) = qu;
                u(p,t)  = qu;
                
            end
        else
            
            % there is only one policy:
            %--------------------------------------------------------------
            pu       = 1;                  % empirical prior 
            qu       = 1;                  % posterior      
        end
        
        % Bayesian model averaging of hidden states (over policies)
        %------------------------------------------------------------------
        for f = 1:Nf
            for i = 1:S
                X{f}(:,i) = reshape(x{f}(:,i,:),Ns(f),Np)*u(:,t);
            end
        end
        
        % processing (i.e., reaction) time
        %----------------------------------------------------------------------
        rt(t)      = toc(tstart);
        
        % record (negative) free energies
        %------------------------------------------------------------------
        MDP.F(:,t) = F;
        MDP.G(:,t) = Q;
        MDP.H(1,t) = qu'*MDP.F(p,t) - qu'*(log(qu) - log(pu));
        
        % check for residual uncertainty in hierarchical schemes
        %------------------------------------------------------------------
        if isfield(MDP,'factor')
            
            for f = MDP.factor(:)'
                qx   = X{f}(:,1);
                H(f) = qx'*spm_log(qx);
            end
            
            % break if there is no further uncertainty to resolve
            %--------------------------------------------------------------
            if sum(H) > - chi
                T = t;
            end
        end
        
        
        % action selection and sampling of next state (outcome)
        %==================================================================
        if t < T
            
            % marginal posterior probability of action (for each modality)
            %--------------------------------------------------------------
            Pu    = zeros([Nu,1]);
            for i = 1:Np
                sub        = num2cell(V(t,i,:));
                Pu(sub{:}) = Pu(sub{:}) + u(i,t);
            end
            
            % action selection - a softmax function of action potential
            %--------------------------------------------------------------
            sub         = repmat({':'},1,Nf);
            Pu(:)       = spm_softmax(alpha*log(Pu(:)));
            P(sub{:},t) = Pu;
            
            % next action - sampled from marginal posterior
            %--------------------------------------------------------------
            try
                a(:,t)  = MDP.u(:,t);
            catch
                ind     = find(rand < cumsum(Pu(:)),1);
                a(:,t)  = spm_ind2sub(Nu,ind);
            end
            
            
            % update policy and states for moving policies
            %--------------------------------------------------------------
            if isfield(MDP,'U')
                
                for f = 1:Nf
                    V(t,:,f) = a(f,t);
                end
                for j = 1:size(MDP.U,1)
                    if (t + j) < T
                        V(t + j,:,:) = MDP.U(j,:,:);
                    end
                end
                
                % and reinitialise expectations about hidden states
                %----------------------------------------------------------
                for f = 1:Nf
                    for k = 1:Np
                        x{f}(:,:,k) = 1/Ns(f);
                    end
                end
            end
            
        elseif t == T
            break;
        end
    end
end

% learning � accumulate concentration parameters
%==========================================================================
for t = 1:T
    
    % mapping from hidden states to outcomes: a
    %----------------------------------------------------------------------
    if isfield(MDP,'a')
        for g = 1:Ng
            da     = sparse(o(g,t),1,1,No(g),1);
            for  f = 1:Nf
                da = spm_cross(da,X{f}(:,t));
            end
            da       = da.*(MDP.a{g} > 0);
            MDP.a{g} = MDP.a{g} + da*eta;
        end
    end
    
    % mapping from hidden states to hidden states: b(u)
    %----------------------------------------------------------------------
    if isfield(MDP,'b') && t > 1
        for f = 1:Nf
            b     = MDP.b{f};
            for k = 1:Np
                v   = V(t - 1,k,f);
                db  = u(k,t)*x{f}(:,t,k)*x{f}(:,t - 1,k)';
                db  = db.*(MDP.b{f}(:,:,v) > 0);
                MDP.b{f}(:,:,v) = MDP.b{f}(:,:,v) + db*eta;
            end
        end
    end
    
    % accumulation of prior preferences: (c)
    %----------------------------------------------------------------------
    if isfield(MDP,'c')
        for g = 1:Ng
            dc = sparse(o(g,t),1,1,No(g),1);
            if size(MDP.c{g},2)>1
                dc = dc.*(MDP.c{g}(:,t)>0);
                MDP.c{g}(:,t) = MDP.c{g}(:,t) + dc*eta;
            else
                dc = dc.*(MDP.c{g}>0);
                MDP.c{g} = MDP.c{g} + dc*eta;
            end
        end
    end
end

% initial hidden states:
%--------------------------------------------------------------------------
if isfield(MDP,'d')
    for f = 1:Nf
        i = MDP.d{f} > 0;
        MDP.d{f}(i) = MDP.d{f}(i) + X{f}(i,1);
    end
end

% policies
%--------------------------------------------------------------------------
if isfield(MDP,'e')
    MDP.e = MDP.e + u(:,T);
end

% (negative) free energy of parameters (i.e., complexity)
%--------------------------------------------------------------
for g = 1:Ng
    if isfield(MDP,'a')
        da        = MDP.a{g} - pA{g};
        MDP.Fa(g) = sum(spm_vec(spm_betaln(MDP.a{g}))) - ...
                    sum(spm_vec(spm_betaln(pA{g})))    - ...
                    spm_vec(da)'*spm_vec(qA{g});
    end
    
    if isfield(MDP,'c')
        dc        = MDP.c{g} - pC{g};
        MDP.Fc(g) = sum(spm_vec(spm_betaln(MDP.c{g}))) - ...
                    sum(spm_vec(spm_betaln(pC{g})))    - ...
                    spm_vec(dc)'*spm_vec(Vo{g});
    end
end

for f = 1:Nf
    if isfield(MDP,'b')
        db        = MDP.b{f} - pB{f};
        MDP.Fb(f) = sum(spm_vec(spm_betaln(MDP.b{f}))) - ...
                    sum(spm_vec(spm_betaln(pB{f})))    - ...
                    spm_vec(db)'*spm_vec(qB{f});
    end
    if isfield(MDP,'d')
        dd        = MDP.d{f} - pD{f};
        MDP.Fd(f) = sum(spm_vec(spm_betaln(MDP.d{f}))) - ...
                    sum(spm_vec(spm_betaln(pD{f})))    - ...
                    spm_vec(dd)'*spm_vec(qD{f});
    end
end    

if isfield(MDP,'e')
    de     = MDP.e - pE;
    MDP.Fe = sum(spm_vec(spm_betaln(MDP.e))) - ...
             sum(spm_vec(spm_betaln(pE)))    - ...
             spm_vec(de)'*spm_vec(qE);
end

% simulated dopamine (or cholinergic) responses
%--------------------------------------------------------------------------
if Np > 1
    dn = 8*gradient(wn) + wn/8;
else
    dn = [];
    wn = [];
end

% Bayesian model averaging of expected hidden states over policies
%--------------------------------------------------------------------------
for f = 1:Nf
    Xn{f} = zeros(Ni,Ns(f),T,T);
    Vn{f} = zeros(Ni,Ns(f),T,T);
    for i = 1:T
        for k = 1:Np
            Xn{f}(:,:,:,i) = Xn{f}(:,:,:,i) + xn{f}(:,:,1:T,i,k)*u(k,i);
            Vn{f}(:,:,:,i) = Vn{f}(:,:,:,i) + vn{f}(:,:,1:T,i,k)*u(k,i);
        end
    end
end

% use penultimate beliefs about moving policies
%--------------------------------------------------------------------------
if isfield(MDP,'U')
    u(:,T)  = [];
    try un(:,(end - Ni + 1):end) = []; catch, end
end

% assemble results and place in NDP structure
%--------------------------------------------------------------------------
MDP.T   = T;              % number of belief updates
MDP.V   = V;              % policies
MDP.P   = P;              % probability of action at time 1,...,T - 1
MDP.Q   = x;              % conditional expectations over N hidden states
MDP.R   = u;              % conditional expectations over policies
MDP.X   = X;              % Bayesian model averages over T outcomes
MDP.C   = Vo;             % utility

if HMM, return, end

MDP.o   = o;              % outcomes at 1,...,T
MDP.s   = s;              % states   at 1,...,T
MDP.u   = a;              % action   at 1,...,T - 1
MDP.w   = gu;             % posterior expectations of precision (policy)

MDP.un  = un;             % simulated neuronal encoding of policies
MDP.vn  = Vn;             % simulated neuronal prediction error
MDP.xn  = Xn;             % simulated neuronal encoding of hidden states
MDP.wn  = wn;             % simulated neuronal encoding of precision
MDP.dn  = dn;             % simulated dopamine responses (deconvolved)
MDP.rt  = rt;             % simulated reaction time (seconds)


% plot
%==========================================================================
if OPTIONS.plot
    if ishandle(OPTIONS.plot)
        figure(OPTIONS.plot); clf
    else
        spm_figure('GetWin','MDP'); clf
    end
    spm_MDP_VB_trial(MDP)
end

function A = spm_log(A)
% log of numeric array plus a small constant
%--------------------------------------------------------------------------
A  = log(A + 1e-16);


function A = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
for i = 1:size(A,2)
    for j = 1:size(A,3)
        for k = 1:size(A,4)
            for l = 1:size(A,5)
                S = sum(A(:,i,j,k,l),1);
                if S > 0
                    A(:,i,j,k,l) = A(:,i,j,k,l)/S;
                else
                    A(:,i,j,k,l) = 1/size(A,1);
                end
            end
        end
    end
end

function A = spm_back(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
for i = 1:size(A,1)
    A(i,:,:,:,:) = A(i,:,:,:,:)/sum(A(i,:));
end

function A = spm_cum(A)
% summation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
for i = 1:size(A,2)
    for j = 1:size(A,3)
        for k = 1:size(A,4)
            for l = 1:size(A,5)
                A(:,i,j,k,l) = sum(A(:,i,j,k,l),1);
            end
        end
    end
end

function A = spm_psi(A)
% normalisation of a probability transition rate matrix (columns)
%--------------------------------------------------------------------------
for i = 1:size(A,2)
    for j = 1:size(A,3)
        for k = 1:size(A,4)
            for l = 1:size(A,5)
                A(:,i,j,k,l) = psi(A(:,i,j,k,l)) - psi(sum(A(:,i,j,k,l)));
            end
        end
    end
end

function C = spm_joint(A,B)
% subscripts from linear index
%--------------------------------------------------------------------------
C = zeros(size(A,1),size(A,2),size(B,2));
for i = 1:size(A,1)
    C(i,:,:) = spm_cross(A(i,:),B(:,i));
end
C = spm_norm(C);


function sub = spm_ind2sub(siz,ndx)
% subscripts from linear index
%--------------------------------------------------------------------------
n = numel(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1,
    vi       = rem(ndx - 1,k(i)) + 1;
    vj       = (ndx - vi)/k(i) + 1;
    sub(i,1) = vj;
    ndx      = vi;
end



