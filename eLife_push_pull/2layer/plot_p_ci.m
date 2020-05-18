for i=1:20 
N=10000;
p_a=2;
p_b=10;
p_c=100;
b1=0.25;
b2=0.15;
tau=5;
B=1;
[i]=deal(0);
[xi_a,xi_b,xi_c] = HirPatterns(N,p_a,p_b,p_c,b1,b2);
xi_a=reshape(xi_a,[N,p_a]);
xi_b=reshape(xi_b,[N,p_a*p_b]);
xi_b=(xi_b-0.5)*2;
xi_c=reshape(xi_c,[N,p_a*p_b*p_c]);
xi_c=(xi_c-0.5)*2;
c_i=b1^2.*xi_c(:,1).*(sum(xi_c(:,2:p_c),2));
c_ii = (b1^2.*xi_c(:,1).*(sum(xi_c(:,2:p_c),2))-b1*xi_c(:,1).*xi_b(:,1));
edges = [-4:0.25:4];
figure()
h_i = histogram(c_i,edges);
hold on
h_ii = histogram(c_ii,edges);
h_i.NumBins= h_ii.NumBins;
end