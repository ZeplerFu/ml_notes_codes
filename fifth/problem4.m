n = 5;
psis = cell(n-1, 1);
for i = 1:(n-1)
psis{i} = rand(2,2);
end
[ ma ] = JCT4MarkovChain( psis );
ptest = cell(4,1);
ptest{1} = [0.1, 0.7; 0.8, 0.3];
ptest{2} = [0.5, 0.1; 0.1, 0.5];
ptest{3} = [0.1, 0.5; 0.5, 0.1];
ptest{4} = [0.9, 0.3; 0.1, 0.3];
[ mtest] = JCT4MarkovChain(ptest);
function[ ma ] = JCT4MarkovChain( po )
ma = po;
n = size(ma,1);
se = ones(n-1,2);
for i = 1:n-1
se(i,:) = sum(ma{i});
ma{i+1} = ma{i+1}.*(se(i,:)'*[1,1]);
end
for i = 1:n-1
sold = se(n-i,:);
se(n-i,:) = sum(ma{n-i+1},2)';
ma{n-i} = ma{n-i}.*([1;1]*(se(n-i,:)./sold));
end
for i = 1:n
ma{i} = ma{i}/sum(sum(ma{i}));
end
end
