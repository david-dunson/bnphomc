function S=stirling(n,k)
%function S=stirling(n,k)
%
% The number of ways of partitioning a set of n elements 
% into k nonempty sets is called 
% a Stirling set number. For example, the set {1,2,3} can 
% be partitioned into three subsets in one way: 
% {{1},{2},{3}}; into two subsets in three ways: 
% {{1,2},{3}}, {{1,3},{2}}, and {{1},{2,3}}; and into 
% one subset in one way: {{1,2,3}}.
S=0;
for i=0:k,
    S=S+(-1)^i*factorial(k)/(factorial(i)*factorial(k-i))*(k-i)^n;
end;
S=S/factorial(k);