function buildMex(src_path, include_path)
% BUILDMEX Builds all MEX files in SRC_PATH and place the resulting files 
% in SRC_PATH/../MEXBUILD. The optional INCLUDE_PATH specifies folders to
% search for `#include` files.

if src_path(end) ~= '/', src_path(end+1) = '/'; end

if nargin == 2
    if include_path(end) ~= '/', include_path(end+1) = '/'; end
end    

listing = dir([src_path '*.cpp']);
listing = [listing; dir([src_path '*.c'])];

build_path = [src_path '../mexBuild'];
mkdir(build_path)

for i = 1 : numel(listing)

    filename = listing(i).name;
    if nargin == 2, mex('-outdir', build_path, ['-I' include_path], [src_path filename]);
    else mex('-outdir', build_path, [src_path filename]); end
    
end

