function [ Weight] = Hebb_weight(pattern)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[N,num_pat]=size(pattern);
Weight=zeros(N,N);
mean_pat=mean(mean(pattern,2));
mean_pat=mean_pat*ones(N,num_pat);
% xi_mean=0;
Weight=(pattern-mean_pat)*(pattern-mean_pat)';
Weight=(Weight-diag(diag(Weight)))/N;
Weight=Weight/norm(Weight);

end

