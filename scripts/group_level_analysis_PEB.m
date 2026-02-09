% 
% This script applies the group-level analysis described in the paper
% "Understanding mechanisms of voluntary engagement of mental effort using 
% active inference" of R. Maramotti, T. Parr, M. Tondelli, D. Ballotta, 
% S. Manohar, G. Zamboni, G. Pagnoni.
% Group-level analysis is performed using Parametric Empirical Bayes (PEB,
% comparing values of e and c in RLX and EXR blocks
%
% Riccardo Maramotti
% Thomas Parr
%
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
 
close all
clear

% Define subjects
subjects = [1:3, 5:21];
n_sub = length(subjects);

% Load first level results
load('../results/POMDP estimates/Z_EXR_allSubj.mat');
Z_EXR.Ep = Z.Ep;
Z_EXR.Cp = Z.Cp;
Z_EXR.F = Z.F;

load('../results/POMDP estimates/Z_RLX_allSubj.mat');
Z_RLX.Ep = Z.Ep;
Z_RLX.Cp = Z.Cp;
Z_RLX.F = Z.F;
clear Z

% Load subjects's informations
data_dir = "../data";
df_info = readtable(fullfile(data_dir, "allSubjects_demografic_data.xlsx"));

% Define DCM structure using first model results
for i = 1 : n_sub
    DCM{i,1}.M.pE.c = 0;              % prior c of i-th subj 
    DCM{i,1}.M.pE.e = 0;              % prior e of i-th subj
    DCM{i,1}.M.pE.lambda = log(0.25); % prior lambda of i-th subj
    DCM{i,1}.M.pE.alpha = log(0.5);   % prior alpha of i-th subj
    DCM{i,1}.M.pC = diag([1/4 1/4 1/4 1/4]);

    DCM{i,1}.Ep = Z_EXR.Ep(subjects(i), :);
    DCM{i,1}.Cp = [Z_EXR.Cp(subjects(i), 1), Z_EXR.Cp(subjects(i), 5), Z_EXR.Cp(subjects(i), 6), 0.01; 
                Z_EXR.Cp(subjects(i), 5), Z_EXR.Cp(subjects(i), 2), Z_EXR.Cp(subjects(i), 7), 0.01; 
                Z_EXR.Cp(subjects(i), 6), Z_EXR.Cp(subjects(i), 7), Z_EXR.Cp(subjects(i), 3), 0.01;
                0.01,  0.01, 0.01, Z_EXR.Cp(subjects(i), 4) ];
    DCM{i,1}.F = Z_EXR.F(subjects(i));
end

for i = 1 : n_sub
    DCM{i + n_sub,1}.M.pE.c = 0;                % prior c of i-th subj 
    DCM{i + n_sub,1}.M.pE.e = 0;                % prior e of i-th subj
    DCM{i + n_sub,1}.M.pE.lambda = log(0.25);   % prior lambda of i-th subj
    DCM{i + n_sub,1}.M.pE.alpha = log(0.5);     % prior alpha of i-th subj
    DCM{i + n_sub,1}.M.pC = diag([1/4 1/4 1/4 1/4]);

    DCM{i + n_sub,1}.Ep = Z_RLX.Ep(subjects(i), :);
    DCM{i + n_sub,1}.Cp = [Z_RLX.Cp(subjects(i), 1), Z_RLX.Cp(subjects(i), 5), Z_RLX.Cp(subjects(i), 6), 0.01; 
                Z_RLX.Cp(subjects(i), 5), Z_RLX.Cp(subjects(i), 2), Z_RLX.Cp(subjects(i), 7), 0.01; 
                Z_RLX.Cp(subjects(i), 6), Z_RLX.Cp(subjects(i), 7), Z_RLX.Cp(subjects(i), 3), 0.01;
                0.01, 0.01, 0.01, Z_RLX.Cp(subjects(i), 4) ];
    DCM{i + n_sub,1}.F = Z_RLX.F(subjects(i));
end

% Define second level design matrix
M.X = zeros(2*n_sub, 1+n_sub);
M.X(:,1) = 1;
M.X(1:n_sub,2) = 1;
for i = 1 : n_sub
    M.X(:, i+2) = - 1/(n_sub - 1);
    M.X([i, i+n_sub], i+2) = 1;
end
M.Xnames = ["intercept", "EXR", repmat("sub", 1, n_sub) + string(subjects)];

% Figure of the design matrix
figure, imagesc(M.X); colormap("sky"); colorbar;
ylabel("models"); xlabel("Second level parameters"); title("PEB design matrix");

% Run PEB on the parameters c and e
field = {'c', 'e'};
[PEB, DCM] = spm_dcm_peb(DCM, M, field);
spm_dcm_peb_review(PEB, DCM);

% Re-defne PEB.Ep as a structure
a = PEB.Ep;
PEB.Ep = [];
PEB.Ep.intercept = a(:,1);
PEB.Ep.EXR = a(:,2);
PEB.Ep.sub01 = a(:,3);
PEB.Ep.sub02 = a(:,4);
PEB.Ep.sub03 = a(:,5);
PEB.Ep.sub05 = a(:,6);
PEB.Ep.sub06 = a(:,7);
PEB.Ep.sub07 = a(:,8);
PEB.Ep.sub08 = a(:,9);
PEB.Ep.sub09 = a(:,10);
PEB.Ep.sub10 = a(:,11);
PEB.Ep.sub11 = a(:,12);
PEB.Ep.sub12 = a(:,13);
PEB.Ep.sub13 = a(:,14);
PEB.Ep.sub14 = a(:,15);
PEB.Ep.sub15 = a(:,16);
PEB.Ep.sub16 = a(:,17);
PEB.Ep.sub17 = a(:,18);
PEB.Ep.sub18 = a(:,19);
PEB.Ep.sub19 = a(:,20);
PEB.Ep.sub20 = a(:,21);
PEB.Ep.sub21 = a(:,22);

% Run BMC (bayesian model comparison) on EXR
field = {'EXR'};
PEB2 = my_spm_dcm_bmr(PEB, field);