function fprintfmat(A, every_x_row, every_x_column, precision, specifier)
% Prints a matrix separating rows with '-' and columns with '|'.
%
% Inputs:
% - A                    : the matrix to be printed
% - every_x_row          : number of rows separated by '-'
% - every_x_column       : number of columns separated by '|'
% - precision (optional) : number of digits after a period (integer)
% - specifier (optional) : notation of the output (single character)
%
% Example: fprintf(A,1,2,3,'f')

if rem(size(A,2), every_x_column) ~= 0
    error('Incompatible number of columns.')
end

if rem(size(A,1), every_x_row) ~= 0
    error('Incompatible number of rows.')
end

if nargin == 3
    precision = 3;
    specifier = 'f';
else
    if ~isscalar(precision)
        error('Wrong precision.')
    end

    if ~ischar(specifier)
        error('Wrong specifier.')
    end
end

max_length = length(num2str(max(round(abs(A(:))))));
max_length = max_length + precision + 2;
format = ['% ' num2str(max_length) '.' num2str(precision) specifier];

column_block = [repmat([format ' '], 1, every_x_column) '|'];
str_row = repmat(column_block, 1, size(A, 2) / every_x_column);
str_row(end) = [];
row_delimiter = repmat('-', 1, length(sprintf(str_row, A(1,:))));
row_block = repmat([str_row '\n'], 1, every_x_row);
fprintf(['\n' row_delimiter '\n']);
fprintf([ row_block row_delimiter '\n' ], A);
fprintf('\n');
