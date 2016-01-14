function gray = rgb2gray(rgb)
% RGB2GRAY Converts RGB values to grayscale.

gray = 0.2989 * rgb(:,:,1) + 0.5870 * rgb(:,:,2) + 0.1140 * rgb(:,:,3); 
