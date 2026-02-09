function [mdp, Z] = train_POMDP_model_RLX(p)
% 
% This script inverts the POMDP model in the paper "Understanding mechanisms of
% voluntary engagement of mental effort using active inference" of
% R. Maramotti, T. Parr, M. Tondelli, D. Ballotta, S. Manohar, G. Zamboni,
% G. Pagnoni.
% This model is meant to find c, e, lambda and alpha for each subject.
%
% Riccardo Maramotti
% Thomas Parr
%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

% set up and preliminaries
rng("twister")  % For reproducibility

data_dir = "../data";

try e = exp(p.e);           catch, e = exp(-1.3);   end % Bias towards reading word
try c = exp(p.c);           catch, c = 1;           end         % Preference for being correct
try lambda = exp(p.lambda); catch lambda = 0.25;    end   % Weight for the Shannon entropy
try alpha = p.alpha;        catch alpha = log(0.5);      end   % Weight for the Shannon entropy

W = 1;          % Name colour [1] or read word [2]
N = 25;         % Number of words per block + instructions

% First level POMDP (HMM in this case)
%==========================================================================

label.factor = {'Written word','Word Colour','Task Sequence','Instruction','Response','Correct?'};

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
MDP.alpha  = alpha;
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
mdp.s = [1;W;1];
mdp.link = zeros(numel(mdp.MDP.D),numel(mdp.A));
mdp.link(3,1) = 1;
mdp.link(4,2) = 1;
mdp.link(5,3) = 1;
mdp.link(6,4) = 1;
mdp.label     = label;
mdp.tau       = 8;
mdp           = spm_MDP_check(mdp);
OPTIONS.gamma = 1;

if exist('p'), return, end

% Solve POMDP
%--------------------------------------------------------------------------
MDP = spm_MDP_VB_X(mdp,OPTIONS);


% Model fits
%--------------------------------------------------------------------------
subjects = [1:3, 5:21];

U  = [];
Ep = [];
Cp = [];

for i = subjects

    clear M
    M.L         = @MDP_Stroop_L;
    M.G         = @(p) MDP_Stroop_Gen(p, MDP(1));
    M.pE.c      = 0;                   % prior means (parameters)
    M.pE.e      = 0;
    M.pE.lambda = log(0.25);
    M.pE.alpha  = log(0.5);
    M.pC        = diag([1/4 1/4 1/4 1/4]); % prior variance (parameters)
    M.ch        = 1; % Include choice data
    M.rt        = 1; % Include reaction time data

    df = readtable(fullfile(data_dir, sprintf("sub_%02d/sub_%02d_taskAnswers", i, i)));
    df.keyWord_rt = str2double(df.keyWord_rt);
    Y = func_build_Y_from_data_RLX(df);
    [EP,CP,F] = spm_nlsi_Newton(M,U,Y);

    display(['Inverted model subject ' num2str(i) '/' num2str(length(MDP))])
    Ep(i,:) = spm_vec(EP)';
    disp(Ep(i,:))
    Cp(i,:) = [diag(CP)' CP(1,2) CP(1,3) CP(1,4) CP(2,3) CP(2,4) CP(3,4)];
    disp(Cp(i,:))
    Fvec(i) = F;

    Z.Ep = Ep;
    Z.Cp = Cp;
    Z.F = Fvec;
    save('Z_RLX_allSubj','Z')
end



%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%

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


function L = MDP_Stroop_L(P,M,~,Y)
% Likelihood function for model fitting.

L = 0;
n_blocks = 8;

lambda = exp(P.lambda);
alpha = P.alpha;

% alpha = M.Z_RT.alpha;
% v = M.Z_RT.v;

for j = 1:n_blocks
    mdp(j)   = M.G(P);
end

% Assign outcomes
%--------------------------------------------------------------------------
for i = 1:mdp(1).T                                                         
    for j = 1 : n_blocks
        mdp(j).mdp(i).o = Y.o{i}(:,[2*j-1, 2*j]);
    end
end

% Invert model (forcing it to take same actions)
%--------------------------------------------------------------------------
OPTIONS.gamma = 1;
for j = 1 : n_blocks                                    
    MDP(j) = spm_MDP_VB_X(mdp(j),OPTIONS); 
end

if M.ch == 1
    
% Evaluate likelihood of actions
%--------------------------------------------------------------------------
    for i = 2:length(MDP(1).mdp)
        for b = 1 : n_blocks
            for j = 1:numel(MDP(b).mdp(i).xn)
                xn{j} = MDP(b).mdp(i).xn{j}(end,:,2,1);
            end
            x_temp = spm_softmax(lambda*spm_log(spm_dot(MDP(b).mdp(i).A{4},xn)));
            L = L + spm_log(x_temp(MDP(b).mdp(i).o(4,2)));
        end
    end
    
end

if M.rt == 1
% Evaluate likelihood of reaction times
%--------------------------------------------------------------------------
    R = [];
    for j = 1:n_blocks
        R = [R MDP_Stroop_RT(MDP(j))];
    end
    L = L + sum(log( spm_Npdf(log(Y.r), R' + alpha, 1/16)));
end


function MDP = MDP_Stroop_Gen(p, MDP)
% POMDP for generative model
e = exp(p.e - 1.3);
c = exp(p.c);
MDP.E    = spm_softmax([e;-e]);
MDP.C{4} = [c;-c];


function [Y] = func_build_Y_from_data_RLX(df)
n_stim_per_block = 24;
n_blocks = 8;

% Delete non-RLX trials
df(df.effort == "EXR",:) = [];

% Transform word names in numbers
df.word_num = zeros(size(df,1),1);
df{df.word == "BLU", "word_num"} = 1;
df{df.word == "GIALLO", "word_num"} = 2;
df{df.word == "VERDE", "word_num"} = 3;
df{df.word == "ROSSO", "word_num"} = 4;

% Transform color names in numbers
df.color_num = zeros(size(df,1), 1);
df{df.color == "blue", "color_num"} = 1;
df{df.color == "yellow", "color_num"} = 2;
df{df.color == "greeen", "color_num"} = 3;
df{df.color == "red", "color_num"} = 4;

% Transform task names in numbers
df.task_num = zeros(size(df,1), 1);
df{df.task == "color", "task_num"} = 1;
df{df.task == "word", "task_num"} = 2;

Y.o = {};
for j = 1 : n_blocks

    % Instructions
    df_stim = df((j-1)*n_stim_per_block + 1, :);
    Y.o{1}(:,j*2-1) = [5; ...   % outcome: written word (5 = null)
        5; ...                  % outcome: color (5 = null)
        df_stim.task_num; ...   % outcome: instruction (1 = color, 2 = word)
        5];                     % outcome: button press (5 = null)
    Y.o{1}(:,j*2) = [5; ...     % outcome: written word (5 = null)
        5; ...                  % outcome: color (5 = null)
        df_stim.task_num; ...   % outcome: instruction (1 = color, 2 = word)
        5];                     % outcome: button press (5 = null)

    % All other stimuli
    for k = 1 : n_stim_per_block
        df_stim = df((j-1)*n_stim_per_block + k, :);

        % Reading phase
        Y.o{k+1}(:,j*2-1) = [df_stim.word_num; ...  % outcome: written word
            df_stim.color_num; ...                  % outcome: color
            3; ...                                  % outcome: instruction (3 = null)
            5];                                     % outcome: button press (5 = null)

        % Response phase
        Y.o{k+1}(:,j*2) = [df_stim.word_num; ...    % outcome: written word
            df_stim.color_num; ...                  % outcome: color
            3; ...                                  % outcome: instruction (3 = null)
            df_stim.pressed_key];                   % outcome: button press
    end
end

Y.r = df.keyWord_rt;