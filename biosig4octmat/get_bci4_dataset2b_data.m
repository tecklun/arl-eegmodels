% BIOSIG runs on Matlab and Octave. 
% This is a script installing all components in an automatically.
%  
% 1) extract the files and
% 2) save the BIOSIG files in <your_directory>
% 3) start matlab
%    cd <your_directory>
%    biosig_installer 
% 4) For a permanent installation, save the default path with 
%     PATH2RC or
%     PATHTOOL and click on the "SAVE" button. 
% 5) For removing the toolbox 
%    remove the path to 
%       HOME/tsa
%       HOME/NaN
%       HOME/BIOSIG/ and all its subdirectories
% 
%  NOTE: by default also the NaN-toolbox is installed - 
%  - a statistical toolbox for handling missing values - which 
%  changes the behaviour of some standard functions. For more  
%  information see NaN/README.TXT . In case you do not want this, 
%  you can excluded the path to NaN/*. The BIOSIG tools will still 
%  work, but does not support the handling of NaN's.

% Copyright (C) 2003-2010,2013,2015,2020 by Alois Schloegl <alois.schloegl@gmail.com>
% This is part of the BIOSIG-toolbox http://biosig.sf.net/

BIOSIG_MATLAB_PATH = getenv('BIOSIG_MATLAB_PATH');
if isempty(BIOSIG_MATLAB_PATH)
	if exist('./t200_FileAccess','dir')
		BIOSIG_MATLAB_PATH = pwd;
	else
		fprintf(2,'Error: biosig subdirectories not found\n');
	        return;
	end;
end; 

subdirs={'doc','t200_FileAccess','t210_Events','t250_ArtifactPreProcessingQualityControl','t300_FeatureExtraction','t400_Classification','t330_StimFit','t450_MultipleTestStatistic','t490_EvaluationCriteria','t500_Visualization','t501_VisualizeCoupling'};
addpath(sprintf(fullfile(BIOSIG_MATLAB_PATH,'%s:'),subdirs{:}))

if exist('OCTAVE_VERSION','builtin'),
	try
		pkg load general
	end
	try
		pkg load signal
	end
	try
		pkg load statistics
	end
	try
		pkg load tsa
	end
	try
		pkg load nan
	end
else
	%% Matlab
    BIOSIG_DIR = pwd
	path([BIOSIG_DIR,'/viewer'],path);		% viewer
	path([BIOSIG_DIR,'/viewer/utils'],path);	% viewer
	path([BIOSIG_DIR,'/viewer/help'],path);	% viewer
    path([BIOSIG_DIR,'/t200_FileAccess'],path);	% sload path
    
	path([BIOSIG_MATLAB_PATH,'/tsa'],path);		%  Time Series Analysis
	path([BIOSIG_MATLAB_PATH,'/tsa/inst'],path);		%  Time Series Analysis
	path([BIOSIG_MATLAB_PATH,'/tsa/src'],path);		%  Time Series Analysis

	if exist([BIOSIG_MATLAB_PATH,'/freetb4matlab'],'dir')
		path(path,[BIOSIG_MATLAB_PATH,'/freetb4matlab/oct2mat']);	% some basic functions used in Octave but not available in Matlab
		path(path,[BIOSIG_MATLAB_PATH,'/freetb4matlab/general']);	% some basic functions used in Octave but not available in Matlab
	end

	fprintf(1,'\nThe NaN-toolbox is going to be installed\n');
	fprintf(1,'The NaN-toolbox is a powerful statistical and machine learning toolbox, \nwhich is also able to handle data with missing values.\n');
	fprintf(1,'Typically, samples with NaNs are simply skipped.\n');
	fprintf(1,'If your data contains NaNs, installing the NaN-toolbox will \nmodify the following functions in order to ignore NaNs:\n');
	fprintf(1,'\tcor, corrcoef, cov, geomean, harmmean, iqr, kurtosis, mad, mahal, mean, \n\tmedian, moment, quantile, prctile, skewness, std, var.\n');
	fprintf(1,'If you do not have NaN, the behaviour is the same; if you have NaNs in your data, you will get more often a reasonable result instead of a NaN-result.\n');
	fprintf(1,'If you do not want this behaviour, remove the directory NaN/inst from your path.\n');
	fprintf(1,'Moreover, NaN-provides also a number of other useful functions. Installing NaN-toolbox is recommended.\n\n');

	%% add NaN-toolbox: a toolbox for statistics and machine learning for data with Missing Values
	path([BIOSIG_MATLAB_PATH,'/NaN'],path);
	%% support both types of directory structure
	if exist([BIOSIG_MATLAB_PATH,'/NaN/inst'],'dir')
		path([BIOSIG_MATLAB_PATH,'/NaN/inst'],path);
	end;
	if exist([BIOSIG_MATLAB_PATH,'/NaN/src'],'dir')
		path([BIOSIG_MATLAB_PATH,'/NaN/src'],path);
	end
end


tmp_biosig_helper_directory = pwd;
try
	if ~exist('OCTAVE_VERSION','builtin') && ~ispc,
		mex -setup
	end; 
        if ~ispc && exist([BIOSIG_MATLAB_PATH,'/NaN/src'],'dir');
		cd([BIOSIG_MATLAB_PATH,'/NaN/src']);
	        make
	end;         
catch 
	fprintf(1,'Compilation of Mex-files failed - precompiled binary mex-files are used instead\n'); 
end;
cd(tmp_biosig_helper_directory);
clear tmp_biosig_helper_directory;

try 
    x = betainv(.5, 1, 2);
catch     
    path(path,[BIOSIG_MATLAB_PATH,'/freetb4matlab/statistics/distributions']);	% Octave-Forge statistics toolbox converted with freetb4matlab
    path(path,[BIOSIG_MATLAB_PATH,'/freetb4matlab/statistics/tests']);	% Octave-Forge statistics toolbox converted with freetb4matlab
    disp('statistics/distribution toolbox (betainv) from freetb4matlab added');
end; 

% try 
%     x = mod(1:10,3)'-1;
%     [Pxx,f]=periodogram(x,[],10,100);
%     [b,a] = butter(5,[.08,.096]);
% catch
% %     path(path,[BIOSIG_MATLAB_PATH,'/freetb4matlab/signal'],'-end');	% Octave-Forge signal processing toolbox converted with freetb4matlab
%     path(path,[BIOSIG_MATLAB_PATH,'/freetb4matlab/signal']);	% Octave-Forge signal processing toolbox converted with freetb4matlab
%     disp('signal processing toolbox (butter,periodogram) from freetb4matlab added');
% end; 

% test of installation
fun = {'butter','periodogram','betainv'};
for k = 1:length(fun),
        x = which(fun{k});
        if isempty(x) || strcmp(x,'undefined'),
                fprintf(2,'Function %s is missing\n',upper(fun{k}));
        end;
end;

disp('BIOSIG-toolbox activated');
disp('	If you want BIOSIG permanently installed, use the command SAVEPATH.')
disp('	or use PATHTOOL to select and deselect certain components.')



%% Extract data file to save as .mat for reading by python
fprintf('Loading A01E.gdf\n')
[A01E_s,A01E_HDR]=sload('.\BCICIV_2a_gdf\A01E.gdf');
fprintf('Loading A01T.gdf\n')
[A01T_s,A01T_HDR]=sload('.\BCICIV_2a_gdf\A01T.gdf');

fprintf('Loading A02E.gdf\n')
[A02E_s,A02E_HDR]=sload('.\BCICIV_2a_gdf\A02E.gdf');
fprintf('Loading A02T.gdf\n')
[A02T_s,A02T_HDR]=sload('.\BCICIV_2a_gdf\A02T.gdf');

fprintf('Loading A03E.gdf\n')
[A03E_s,A03E_HDR]=sload('.\BCICIV_2a_gdf\A03E.gdf');
fprintf('Loading A03T.gdf\n')
[A03T_s,A03T_HDR]=sload('.\BCICIV_2a_gdf\A03T.gdf');

fprintf('Loading A04E.gdf\n')
[A04E_s,A04E_HDR]=sload('.\BCICIV_2a_gdf\A04E.gdf');
fprintf('Loading A04T.gdf\n')
[A04T_s,A04T_HDR]=sload('.\BCICIV_2a_gdf\A04T.gdf');

fprintf('Loading A05E.gdf\n')
[A05E_s,A05E_HDR]=sload('.\BCICIV_2a_gdf\A05E.gdf');
fprintf('Loading A05T.gdf\n')
[A05T_s,A05T_HDR]=sload('.\BCICIV_2a_gdf\A05T.gdf');

fprintf('Loading A06E.gdf\n')
[A06E_s,A06E_HDR]=sload('.\BCICIV_2a_gdf\A06E.gdf');
fprintf('Loading A06T.gdf\n')
[A06T_s,A06T_HDR]=sload('.\BCICIV_2a_gdf\A06T.gdf');

fprintf('Loading A07E.gdf\n')
[A07E_s,A07E_HDR]=sload('.\BCICIV_2a_gdf\A07E.gdf');
fprintf('Loading A07T.gdf\n')
[A07T_s,A07T_HDR]=sload('.\BCICIV_2a_gdf\A07T.gdf');

fprintf('Loading A08E.gdf\n')
[A08E_s,A08E_HDR]=sload('.\BCICIV_2a_gdf\A08E.gdf');
fprintf('Loading A08T.gdf\n')
[A08T_s,A08T_HDR]=sload('.\BCICIV_2a_gdf\A08T.gdf');

fprintf('Loading A09E.gdf\n')
[A09E_s,A09E_HDR]=sload('.\BCICIV_2a_gdf\A09E.gdf');
fprintf('Loading A09T.gdf\n')
[A09T_s,A09T_HDR]=sload('.\BCICIV_2a_gdf\A09T.gdf');

%% Save to .mat
save('bci4_dataset2b.mat',...
    'A01E_s', 'A01E_HDR', 'A01T_s', 'A01T_HDR', ...
    'A02E_s', 'A02E_HDR', 'A02T_s', 'A02T_HDR', ...
    'A03E_s', 'A03E_HDR', 'A03T_s', 'A03T_HDR', ...
    'A04E_s', 'A04E_HDR', 'A04T_s', 'A04T_HDR', ...
    'A05E_s', 'A05E_HDR', 'A05T_s', 'A05T_HDR', ...
    'A06E_s', 'A06E_HDR', 'A06T_s', 'A06T_HDR', ...
    'A07E_s', 'A07E_HDR', 'A07T_s', 'A07T_HDR', ...
    'A08E_s', 'A08E_HDR', 'A08T_s', 'A08T_HDR', ...
    'A09E_s', 'A09E_HDR', 'A09T_s', 'A09T_HDR' ...
    )