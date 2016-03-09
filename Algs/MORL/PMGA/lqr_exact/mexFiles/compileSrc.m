% Use this script to compile all the mex sources

fileext = '*.cpp';
listing = dir(fileext);

addpath(genpath('.'))

mkdir('mexBuild')

for i = 1 : numel(listing)

    filename = listing(i).name;
    mex('-outdir', 'mexBuild/', filename);
    
end

