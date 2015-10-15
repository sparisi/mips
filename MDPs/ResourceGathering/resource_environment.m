function [ hasGems, hasGold, hasEnemy ] = resource_environment( )

nrows = 5;
ncols = 5;

hasGems = false(nrows,ncols);
hasGold = false(nrows,ncols);
hasEnemy = false(nrows,ncols);

hasGems(2,5) = true;
hasGold(1,3) = true;
hasEnemy(2,3) = true;
hasEnemy(1,4) = true;

end

