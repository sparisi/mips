function [ treasure, isWhite ] = deep_environment( )

nrows = 11;
ncols = 10;

treasure = zeros(nrows,ncols);
treasure(2,1) = 1;
treasure(3,2) = 2;
treasure(4,3) = 3;
treasure(5,4) = 5;
treasure(5,5) = 8;
treasure(5,6) = 16;
treasure(8,7) = 24;
treasure(8,8) = 50;
treasure(10,9) = 74;
treasure(11,10) = 124;

% Map to distinguish between white and black cells
isWhite = true(nrows,ncols);
for i = 3 : nrows
    for j = 1 : i - 2
        isWhite(i,j) = false;
    end
end
isWhite(6,5) = false;
isWhite(6,6) = false;
isWhite(7,6) = false;
isWhite(9,8) = false;

end

