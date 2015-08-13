% Use this wrapper to compile all the mex files

fileext = '*.cpp';
listing = dir(fileext);

addpath(genpath('.'))

for i = 1 : numel(listing)

    filename = listing(i).name;
    mex(filename);
    
end
