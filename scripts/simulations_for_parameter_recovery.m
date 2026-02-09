function [mdp] = simulations_before_parameter_recovery()
% 
% This script generates simulated datasets for each subject using
% a hierarchical POMDP model; the sumulated data are necessary for the subsequent
% parameter-recovery analyses described the paper
% "Understanding mechanisms of voluntary engagement of mental effort using
% active inference" of R. Maramotti, T. Parr, M. Tondelli, D. Ballotta,
% S. Manohar, G. Zamboni, G. Pagnoni.
%
% Riccardo Maramotti
% Thomas Parr
%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

clear
close all

% set up and preliminaries
rng("default")  % For reproducibility

subjects = [1:3, 5:21];
N = 25;         % Number of words per block
N_task = N-1;
nRows = N_task*4;    % Number of blocks

for subj = subjects

    df_output = table( ...
        strings(nRows,1), ...            % effort
        strings(nRows,1), ...            % task
        strings(nRows,1), ...            % word
        strings(nRows,1), ...            % color
        strings(nRows,1), ...            % pressed_key
        NaN(nRows,1), ...                % keyWord_rt
        'VariableNames', ...
        ["effort", "task", "word", "color", "pressed_key", "keyWord_rt"]);
    df_output.participant(:) = subj;

    %% RLX
    load('../results/POMDP estimates/Z_RLX_allSubj.mat');
    
    c = exp(Z.Ep(subj, 1));
    e = exp(Z.Ep(subj, 2) - 1.3);
    lambda = exp(Z.Ep(subj, 3));
    alpha = Z.Ep(subj, 4);
    v = log(1/16);
    
    % First level POMDP (HMM in this case)
    %==========================================================================
    
    label.factor = {'Written word','Word Colour','Task Sequence','Instruction','Response','Correct?'};
    
    
    clear A B C D MDP
    
    % Priors over initial states P(s{j} = i, t = 1) = D{j}(i)
    %--------------------------------------------------------------------------
    D{1} = ones(4,1); % Written word:  Red, green, blue, yellow
    D{2} = ones(4,1); % Word colour:   Red, green, blue, yellow
    D{3} = ones(3,1); % Task sequence: Instruction, null, response
    D{4} = ones(2,1); % Instruction:   Colour, written
    D{5} = ones(2,1); % Response:      Colour, written
    D{6} = ones(2,1); % Correct?:      Correct, incorrect
    
    % Transition probabilities P(s{j} = i, t+1|s{j} = k, u = m, t) = B{j}(i,k,m)
    %--------------------------------------------------------------------------
    B{1} = eye(4);    % Written word does not change during trial
    B{2} = eye(4);    % Word colour does not change during trial
    B{3} = [1 0 0;    % Stay in instruction phase if started there
            0 0 0;    % Move to response if in null phase
            0 1 1];   % Stay in response phase if there already
    B{4} = eye(2);    % Instructions do not change throughout trial
    B{5} = eye(2);    % Response does not change throughout trial
    B{5}(:,:,2) = eye(2); % Cheat to get around issue in HMM code
    B{6} = eye(2);    % Correctness of response does not change
    
    % Likelihood (P(o{m} = i|s{n1} = j1, s{n2} = j2...) = A{m}(i,j1,j2,...)
    %--------------------------------------------------------------------------
    for f1 = 1:length(D{1})
        for f2 = 1:length(D{2})
            for f3 = 1:length(D{3})
                for f4 = 1:length(D{4})
                    for f5 = 1:length(D{5})
                        for f6 = 1:length(D{6})
                            % Written/colour of word, and instruction:
                            %--------------------------------------------------
                            if f3 == 1
                                A{1}(5,f1,f2,f3,f4,f5,f6)  = 1; % No word during instruction
                                A{2}(5,f1,f2,f3,f4,f5,f6)  = 1;
                                A{3}(f4,f1,f2,f3,f4,f5,f6) = 1; % Instruction given
                            else
                                A{1}(f1,f1,f2,f3,f4,f5,f6) = 1; % Otherwise, depends on written word factor
                                A{2}(f2,f1,f2,f3,f4,f5,f6) = 1; % or word colour factor
                                A{3}(3,f1,f2,f3,f4,f5,f6)  = 1; % No instruction given
                            end
                            
                            % Verbal response
                            %--------------------------------------------------
                            if f3 == 3
                                if f5 == 1
                                    A{4}(f2,f1,f2,f3,f4,f5,f6) = 1; % Expect to report colour
                                else
                                    A{4}(f1,f1,f2,f3,f4,f5,f6) = 1; % Expect to report written word
                                end
                            else
                                A{4}(5,f1,f2,f3,f4,f5,f6) = 1;      % Expect null word
                            end
                        end
                    end
                end
            end
        end
    end
    
    % Preferences log(P(o{j} = i|C)) = C{j}(i) + const.
    %--------------------------------------------------------------------------
    for g = 1:numel(A)
        C{g} = zeros(size(A{g},1),1);
    end
    
    % Self-generated outcomes (verbal response)
    %--------------------------------------------------------------------------
    n      = zeros(numel(A),2);
    n(4,:) = 1;
    
    % Compile POMDP
    %--------------------------------------------------------------------------
    MDP.T       = 2;
    MDP.A       = A;
    MDP.B       = B;
    MDP.C       = C;
    MDP.D       = D;
    MDP.n       = n;
    MDP.tau     = 8;
    MDP.chi     = -exp(64);
    MDP.label   = label;
    MDP.lambda  = lambda;
    MDP         = spm_MDP_check(MDP);
    
    clear A B C D n label
    
    % Second level POMDP
    %==========================================================================
    
    label.factor = {'Narrative','Instruction','Response'};
    
    % Priors over initial states P(s{j} = i, t = 1) = D{j}(i)
    %--------------------------------------------------------------------------
    D{1} = [1 0]'; % Narrative: instruction, response
    D{2} = [1 1]'; % Instruction: colour, written
    D{3} = [1 1]'; % Response: colour, written
    
    % Transition probabilities P(s{j} = i, t+1|s{j} = k, u = m, t) = B{j}(i,k,m)
    %--------------------------------------------------------------------------
    B{1} = [0 0; 1 1]; % Once instruction has been given, responses
    B{2} = eye(2);     % Instruction is constant over time
    B{3} = zeros(2,2,2);
    B{3}(1,:,2) = 1;   % Choose to respond with colour of word
    B{3}(2,:,1) = 1;   % Choose to respond by reading written word
    
    % Likelihood probabilities P(s{j} = i, t+1|s{j} = k, u = m, t) = B{j}(i,k,m)
    %--------------------------------------------------------------------------
    for f1 = 1:length(D{1})
        for f2 = 1:length(D{2})
            for f3 = 1:length(D{3})
                % Predicted task sequence (factor 3 lower level)
                %--------------------------------------------------------------
                if f1 == 1
                    A{1}(1,f1,f2,f3) = 1;
                else
                    A{1}(2,f1,f2,f3) = 1;
                end
                A{1}(3,f1,f2,f3) = 0;
                
                % Predicted instruction (factor 4 at lower level)
                %--------------------------------------------------------------
                A{2}(f2,f1,f2,f3) = 1;
                
                % Predicted response (factor 5 at lower level)
                %--------------------------------------------------------------
                A{3}(f3,f1,f2,f3) = 1;
                
                % Predicted correct? (factor 6 at lower level)
                %--------------------------------------------------------------
                if f2 == f3
                    A{4}(1,f1,f2,f3) = 1;
                else
                    A{4}(2,f1,f2,f3) = 1;
                end
            end
        end
    end
    
    % Preferences log(P(o{j} = i|C)) = C{j}(i) + const.
    %--------------------------------------------------------------------------
    for g = 1:numel(A)
        C{g} = zeros(size(A{g},1),1);
    end
    C{4} = [c;-c]; % Prefers to be correct
    
    % Policies
    %--------------------------------------------------------------------------
    E    = spm_softmax([e;-e]);
    
    % Compile POMDP structure
    %--------------------------------------------------------------------------
    mdp.MDP = MDP; clear MDP
    
    mdp.A = A;
    mdp.B = B;
    mdp.C = C;
    mdp.D = D;
    mdp.E = E;
    mdp.T = N;
    mdp.link = zeros(numel(mdp.MDP.D),numel(mdp.A));
    mdp.link(3,1) = 1;
    mdp.link(4,2) = 1;
    mdp.link(5,3) = 1;
    mdp.link(6,4) = 1;
    mdp.label     = label;
    mdp.tau       = 8;
    mdp           = spm_MDP_check(mdp);
    OPTIONS.gamma = 1;
    
    % % Solve POMDP for color
    mdp.s = [1;1;1];
    MDP1  = spm_MDP_VB_X(mdp);
    
    [stim1, resp1] = MDP_Stroop_SR(MDP1);
    H1 = zeros(3,numel(resp1));
    for i = 1:numel(resp1)
        H1(1,i) = strcmp(stim1.color{i},stim1.word{i});      % Congruent?
        H1(2,i) = strcmp(['"' stim1.color{i} '"'],resp1{i}); % Correct?
    end
    H1(3,:) = MDP_Stroop_RT(MDP1);                         % Reaction time
    
    df_output.pressed_key(1:N_task) = resp1;
    df_output.word(1:N_task) = stim1.word;
    df_output.color(1:N_task) = stim1.color;
    df_output.effort(1:N_task) = "RLX";
    df_output.task(1:N_task) = "color";
    df_output.keyWord_rt(1:N_task) = exp(alpha + H1(3,:) + randn(1, N_task)*exp(v));
    
    % % Solve POMDP for word
    mdp.s = [1;2;1];
    MDP2  = spm_MDP_VB_X(mdp);
    
    [stim2, resp2] = MDP_Stroop_SR(MDP2);
    H2 = zeros(3,numel(resp2));
    for i = 1:numel(resp2)
        H2(1,i) = strcmp(stim2.color{i},stim2.word{i});      % Congruent?
        H2(2,i) = strcmp(['"' stim2.word{i} '"'],resp2{i});  % Correct?
    end
    H2(3,:) = MDP_Stroop_RT(MDP2);                         % Reaction time
    
    df_output.pressed_key(N_task+1:2*N_task) = resp2;
    df_output.word(N_task+1:2*N_task) = stim2.word;
    df_output.color(N_task+1:2*N_task) = stim2.color;
    df_output.effort(N_task+1:2*N_task) = "RLX";
    df_output.task(N_task+1:2*N_task) = "word";
    df_output.keyWord_rt(N_task+1:2*N_task) = exp(alpha + H2(3,:) + randn(1, N_task)*exp(v));
    
    
    %% EXR
    load('../results/POMDP estimates/Z_EXR_allSubj.mat');
    
    c = exp(Z.Ep(subj, 1));
    e = exp(Z.Ep(subj, 2) - 1.3);
    lambda = exp(Z.Ep(subj, 3));
    alpha = Z.Ep(subj, 4);
    v = log(1/16);
    
    % First level POMDP (HMM in this case)
    %==========================================================================
    
    label.factor = {'Written word','Word Colour','Task Sequence','Instruction','Response','Correct?'};
    
    
    clear A B C D MDP
    
    % Priors over initial states P(s{j} = i, t = 1) = D{j}(i)
    %--------------------------------------------------------------------------
    D{1} = ones(4,1); % Written word:  Red, green, blue, yellow
    D{2} = ones(4,1); % Word colour:   Red, green, blue, yellow
    D{3} = ones(3,1); % Task sequence: Instruction, null, response
    D{4} = ones(2,1); % Instruction:   Colour, written
    D{5} = ones(2,1); % Response:      Colour, written
    D{6} = ones(2,1); % Correct?:      Correct, incorrect
    
    % Transition probabilities P(s{j} = i, t+1|s{j} = k, u = m, t) = B{j}(i,k,m)
    %--------------------------------------------------------------------------
    B{1} = eye(4);    % Written word does not change during trial
    B{2} = eye(4);    % Word colour does not change during trial
    B{3} = [1 0 0;    % Stay in instruction phase if started there
            0 0 0;    % Move to response if in null phase
            0 1 1];   % Stay in response phase if there already
    B{4} = eye(2);    % Instructions do not change throughout trial
    B{5} = eye(2);    % Response does not change throughout trial
    B{5}(:,:,2) = eye(2); % Cheat to get around issue in HMM code
    B{6} = eye(2);    % Correctness of response does not change
    
    % Likelihood (P(o{m} = i|s{n1} = j1, s{n2} = j2...) = A{m}(i,j1,j2,...)
    %--------------------------------------------------------------------------
    for f1 = 1:length(D{1})
        for f2 = 1:length(D{2})
            for f3 = 1:length(D{3})
                for f4 = 1:length(D{4})
                    for f5 = 1:length(D{5})
                        for f6 = 1:length(D{6})
                            % Written/colour of word, and instruction:
                            %--------------------------------------------------
                            if f3 == 1
                                A{1}(5,f1,f2,f3,f4,f5,f6)  = 1; % No word during instruction
                                A{2}(5,f1,f2,f3,f4,f5,f6)  = 1;
                                A{3}(f4,f1,f2,f3,f4,f5,f6) = 1; % Instruction given
                            else
                                A{1}(f1,f1,f2,f3,f4,f5,f6) = 1; % Otherwise, depends on written word factor
                                A{2}(f2,f1,f2,f3,f4,f5,f6) = 1; % or word colour factor
                                A{3}(3,f1,f2,f3,f4,f5,f6)  = 1; % No instruction given
                            end
                            
                            % Verbal response
                            %--------------------------------------------------
                            if f3 == 3
                                if f5 == 1
                                    A{4}(f2,f1,f2,f3,f4,f5,f6) = 1; % Expect to report colour
                                else
                                    A{4}(f1,f1,f2,f3,f4,f5,f6) = 1; % Expect to report written word
                                end
                            else
                                A{4}(5,f1,f2,f3,f4,f5,f6) = 1;      % Expect null word
                            end
                        end
                    end
                end
            end
        end
    end
    
    % Preferences log(P(o{j} = i|C)) = C{j}(i) + const.
    %--------------------------------------------------------------------------
    for g = 1:numel(A)
        C{g} = zeros(size(A{g},1),1);
    end
    
    % Self-generated outcomes (verbal response)
    %--------------------------------------------------------------------------
    n      = zeros(numel(A),2);
    n(4,:) = 1;
    
    % Compile POMDP
    %--------------------------------------------------------------------------
    MDP.T       = 2;
    MDP.A       = A;
    MDP.B       = B;
    MDP.C       = C;
    MDP.D       = D;
    MDP.n       = n;
    MDP.tau     = 8;
    MDP.chi     = -exp(64);
    MDP.label   = label;
    MDP.lambda  = lambda;
    MDP         = spm_MDP_check(MDP);
    
    clear A B C D n label
    
    % Second level POMDP
    %==========================================================================
    
    label.factor = {'Narrative','Instruction','Response'};
    
    % Priors over initial states P(s{j} = i, t = 1) = D{j}(i)
    %--------------------------------------------------------------------------
    D{1} = [1 0]'; % Narrative: instruction, response
    D{2} = [1 1]'; % Instruction: colour, written
    D{3} = [1 1]'; % Response: colour, written
    
    % Transition probabilities P(s{j} = i, t+1|s{j} = k, u = m, t) = B{j}(i,k,m)
    %--------------------------------------------------------------------------
    B{1} = [0 0; 1 1]; % Once instruction has been given, responses
    B{2} = eye(2);     % Instruction is constant over time
    B{3} = zeros(2,2,2);
    B{3}(1,:,2) = 1;   % Choose to respond with colour of word
    B{3}(2,:,1) = 1;   % Choose to respond by reading written word
    
    % Likelihood probabilities P(s{j} = i, t+1|s{j} = k, u = m, t) = B{j}(i,k,m)
    %--------------------------------------------------------------------------
    for f1 = 1:length(D{1})
        for f2 = 1:length(D{2})
            for f3 = 1:length(D{3})
                % Predicted task sequence (factor 3 lower level)
                %--------------------------------------------------------------
                if f1 == 1
                    A{1}(1,f1,f2,f3) = 1;
                else
                    A{1}(2,f1,f2,f3) = 1;
                end
                A{1}(3,f1,f2,f3) = 0;
                
                % Predicted instruction (factor 4 at lower level)
                %--------------------------------------------------------------
                A{2}(f2,f1,f2,f3) = 1;
                
                % Predicted response (factor 5 at lower level)
                %--------------------------------------------------------------
                A{3}(f3,f1,f2,f3) = 1;
                
                % Predicted correct? (factor 6 at lower level)
                %--------------------------------------------------------------
                if f2 == f3
                    A{4}(1,f1,f2,f3) = 1;
                else
                    A{4}(2,f1,f2,f3) = 1;
                end
            end
        end
    end
    
    % Preferences log(P(o{j} = i|C)) = C{j}(i) + const.
    %--------------------------------------------------------------------------
    for g = 1:numel(A)
        C{g} = zeros(size(A{g},1),1);
    end
    C{4} = [c;-c]; % Prefers to be correct
    
    % Policies
    %--------------------------------------------------------------------------
    E    = spm_softmax([e;-e]);
    
    % Compile POMDP structure
    %--------------------------------------------------------------------------
    mdp.MDP = MDP; clear MDP
    
    mdp.A = A;
    mdp.B = B;
    mdp.C = C;
    mdp.D = D;
    mdp.E = E;
    mdp.T = N;
    mdp.link = zeros(numel(mdp.MDP.D),numel(mdp.A));
    mdp.link(3,1) = 1;
    mdp.link(4,2) = 1;
    mdp.link(5,3) = 1;
    mdp.link(6,4) = 1;
    mdp.label     = label;
    mdp.tau       = 8;
    mdp           = spm_MDP_check(mdp);
    OPTIONS.gamma = 1;
    
    % % Solve POMDP for color
    mdp.s = [1;1;1];
    MDP1  = spm_MDP_VB_X(mdp);
    
    [stim1, resp1] = MDP_Stroop_SR(MDP1);
    H1 = zeros(3,numel(resp1));
    for i = 1:numel(resp1)
        H1(1,i) = strcmp(stim1.color{i},stim1.word{i});      % Congruent?
        H1(2,i) = strcmp(['"' stim1.color{i} '"'],resp1{i}); % Correct?
    end
    H1(3,:) = MDP_Stroop_RT(MDP1);                         % Reaction time
    
    df_output.pressed_key(2*N_task+1:3*N_task) = resp1;
    df_output.word(2*N_task+1:3*N_task) = stim1.word;
    df_output.color(2*N_task+1:3*N_task) = stim1.color;
    df_output.effort(2*N_task+1:3*N_task) = "EXR";
    df_output.task(2*N_task+1:3*N_task) = "color";
    df_output.keyWord_rt(2*N_task+1:3*N_task) = exp(alpha + H1(3,:) + randn(1, N_task)*exp(v));
    
    % % Solve POMDP for word
    mdp.s = [1;2;1];
    MDP2  = spm_MDP_VB_X(mdp);
    
    [stim2, resp2] = MDP_Stroop_SR(MDP2);
    H2 = zeros(3,numel(resp2));
    for i = 1:numel(resp2)
        H2(1,i) = strcmp(stim2.color{i},stim2.word{i});      % Congruent?
        H2(2,i) = strcmp(['"' stim2.word{i} '"'],resp2{i});  % Correct?
    end
    H2(3,:) = MDP_Stroop_RT(MDP2);                         % Reaction time
    
    df_output.pressed_key(3*N_task+1:4*N_task) = resp2;
    df_output.word(3*N_task+1:4*N_task) = stim2.word;
    df_output.color(3*N_task+1:4*N_task) = stim2.color;
    df_output.effort(3*N_task+1:4*N_task) = "EXR";
    df_output.task(3*N_task+1:4*N_task) = "word";
    df_output.keyWord_rt(3*N_task+1:4*N_task) = exp(alpha + H2(3,:) + randn(1, N_task)*exp(v));

    %% Write simulated tassk answers dataset
    df_output.pressed_key = replace(df_output.pressed_key, '"', '');
    writetable(df_output, sprintf("../results/recovery analysis/sub_%02d_SimulatedTaskAnswers.csv", subj))
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [stim, resp] = MDP_Stroop_animation(MDP)
% This routine returns the stimuli and response sequence from a solved 
% POMDP problem, along with an animation of that sequence.

str = {'red','green','blue','yellow',' '};
col = {[1,0,0],[0 1 0],[0 0 1],[1 1 0]};

if isfield(MDP,'mdp')
    stim.word  = [];
    stim.color = [];
    resp       = [];
    for i = 1:length(MDP.mdp)
        subplot(6,2,1), cla, axis off, title('Stimulus')
        subplot(6,2,2), cla, axis off, title('Response')
        pause(1/64)
        [s, r] = MDP_Stroop_animation(MDP.mdp(i));
        stim.word  = [stim.word, s.word];
        stim.color = [stim.color, s.color];
        resp       = [resp, r];
        if i>1
            subplot(6,2,3)
            text(0.5,0.5-i/4, stim.word(i-1),'Color',stim.color{i-1},'HorizontalAlignment','center','units','normalized','FontSize',10), hold on
            axis off
            subplot(6,2,4)
            text(0.5,0.5-i/4, resp(i-1),'Color','k','HorizontalAlignment','center','units','normalized','FontSize',10), hold on
            axis off
        end
    end
else
    stim.color = [];
    stim.word  = [];
    resp       = [];
    for t = 1:MDP.T
        subplot(6,2,1), cla
        if MDP.o(3,t) == 3
            text(0.5,0.5,str(MDP.o(1,t)),'Color',col{MDP.o(2,t)},'HorizontalAlignment','center','units','normalized','FontSize',12);
            if t == 2, stim.color = [stim.color, {col{MDP.o(2,t)}}]; stim.word = [stim.word, str(MDP.o(1,t))]; end
        else
            if MDP.o(3,t) == 1
                text(0.5,0.5,'Give the colours of the font of the following words','HorizontalAlignment','center','units','normalized','FontSize',10)
            else
                text(0.5,0.5,'Read the words that follow','HorizontalAlignment','center','units','normalized','FontSize',10)
            end
        end
        axis off
        title('Stimulus')
        subplot(6,2,2), cla
        if MDP.o(4,t) ~= 5
            text(0.5,0.5,['"' str{MDP.o(4,t)} '"'],'HorizontalAlignment','center','units','normalized','FontSize',12)
        end
        if t==2 && ~isempty(stim.word), resp  = [resp, {['"' str{MDP.o(4,t)} '"']}]; end
        axis off
        title('Response')
        pause(1/4)
    end
end


function rt = MDP_Stroop_RT(MDP)
% Simulate reaction times (based upon predictive entropy)

for i = 2:length(MDP.mdp)
    x = [];
    for k = 1:size(MDP.mdp(i).xn{1},1)
        for j = 1:numel(MDP.mdp(i).xn)
            xn{j} = MDP.mdp(i).xn{j}(k,:,2,1);
        end
        x(:,end+1) = spm_dot(MDP.mdp(i).A{4},xn);
    end
    v       = -diag(x'*spm_log(x));
    rt(i-1) = v(end);
end

function [stim, resp] = MDP_Stroop_SR(MDP)
% This routine reports simuli and response sequences without generating an
% animation (see above)

str = {'red','green','blue','yellow',' '};
col = {'red','green','blue','yellow'};

if isfield(MDP,'mdp')
    stim.word  = [];
    stim.color = [];
    resp       = [];
    for i = 1:length(MDP.mdp)
        [s, r] = MDP_Stroop_SR(MDP.mdp(i));
        stim.word  = [stim.word, s.word];
        stim.color = [stim.color, s.color];
        resp       = [resp, r];
    end
else
    stim.color = [];
    stim.word  = [];
    resp       = [];
    for t = 1:MDP.T
        if MDP.o(3,t) == 3
            if t == 2, stim.color = [stim.color, {col{MDP.o(2,t)}}]; stim.word = [stim.word, str(MDP.o(1,t))]; end
        end
        if t==2 && ~isempty(stim.word), resp  = [resp, {['"' str{MDP.o(4,t)} '"']}]; end
    end
end
