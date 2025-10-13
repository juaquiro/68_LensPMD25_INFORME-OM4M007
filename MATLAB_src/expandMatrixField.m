function T = expandMatrixField(T, fieldName)
%EXPANDMATRIXFIELD Expand a matrix field of a table into separate columns.
%
%   T = expandMatrixField(T, fieldName) expands the matrix stored in the
%   table variable T.(fieldName) into separate columns. Each new column
%   will be named "fieldName_1", "fieldName_2", ..., with no spaces.
%   The original field is removed from the table.
%
%   Example:
%       % Create a table with a matrix field
%       T = table((1:3)', [1 2 3; 4 5 6; 7 8 9], ...
%                 'VariableNames', {'ID','X'});
%
%       % Expand the matrix field "X" into separate columns
%       T = expandMatrixField(T, 'X');
%
%       % Save to Excel without spaces in headers
%       writetable(T, 'myFile.xlsx');
%
%   See also ARRAY2TABLE, WRITETABLE.

% Extract matrix
M = T.(fieldName);
if ~ismatrix(M)
    error('Field "%s" must be a matrix.', fieldName);
end

% Create new variable names
nCols = size(M,2);
newNames = strcat(fieldName, "_", string(1:nCols));

% Convert to sub-table
Mtable = array2table(M, 'VariableNames', cellstr(newNames));

% Remove original field and append expanded columns
T.(fieldName) = [];
T = [T Mtable];
end
