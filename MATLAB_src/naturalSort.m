function D_sorted = naturalSort(D, prefix)
%NATURALSORT Sort cell array of strings with numeric suffixes in natural order.
%
%   D_sorted = naturalSort(D, prefix) sorts the cell array of strings D
%   according to the numeric value after the specified prefix.
%
%   Input:
%       D      - 1xN cell array of character vectors or strings
%       prefix - string or char specifying the prefix before the number
%
%   Output:
%       D_sorted - 1xN cell array, sorted in natural numeric order
%
%   Example:
%       % Example input
%       D = {'X_1','X_10','X_11','X_2','X_3','X_20'};
%
%       % Sort using prefix "X_"
%       D_sorted = naturalSort(D, 'X_');
%
%       % Result:
%       %   {'X_1','X_2','X_3','X_10','X_11','X_20'}
%
%   See also SORT, CELLFUN, REGEXP, SSCANF.

    % Ensure input is cell array of char
    if isstring(D)
        D = cellstr(D);
    end
    
    % Extract numeric suffix after prefix
    nums = cellfun(@(s) sscanf(strrep(s, prefix, ''), '%d'), D);
    
    % Sort by the numbers
    [~, idx] = sort(nums);
    
    % Apply sorting
    D_sorted = D(idx);
end
