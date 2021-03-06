function [ feedforward ] = Feedforward1(xi_b,xi_c,p_a,p_b)
%UNTITLED3 Summary of this function goes here
%   Xi_b is the primitive matrix before the shape changes
[N,p1]=size(xi_c);
p_c=p1/p_b/p_a;
M=zeros(p_a*p_b,1);
for i=1:p_a*p_b
   for j=1:p_c
       M(i)=M(i)+moverlap(xi_b(:,i),xi_c(:,p_c*(i-1)+j),0)/p_c;
   end
end
xi_b=xi_b./(ones(N,1)*M');
xi_b=reshape(xi_b,[N,1,p_a*p_b]);
Xi_b=repmat(xi_b,[1,p_c,1]);
xi_b=reshape(Xi_b,[N,p1]);
xi_b_mean=mean(mean(xi_b,2));
xi_b_mean=xi_b_mean*ones(N,p1);
xi_c_mean=mean(mean(xi_c,2));
xi_c_mean=xi_c_mean*ones(N,p1);
 
% xi_b_mean=0;
% xi_c_mean=0;
feedforward=(xi_b-xi_b_mean)*(xi_c-xi_c_mean)';
feedforward=(feedforward-diag(diag(feedforward)))/N;
feedforward=feedforward/norm(feedforward);
end