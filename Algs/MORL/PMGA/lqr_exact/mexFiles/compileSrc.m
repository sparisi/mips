% Use this script to compile all the mex sources in the PMGA folder

script_path = mfilename('fullpath');
script_path = script_path(1:end-length(mfilename));

buildMex([script_path 'src/'], [script_path 'include/'])
