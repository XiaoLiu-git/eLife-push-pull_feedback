clear all
a_ext=1;
num_trail=50;
var_size=1:21;
N=1000;
p_a=2;
p_b=4;
p_c=15;
b1_max=0.6;
b2=0.1;
tau=5;
B=1;
lambda1=0.1;
lambda2=0.1;
[m1,m2,m11,m_improv]=deal(zeros(max(var_size),num_trail));
Path=['./results/',datestr(now,6),'/b1'];
mkdir(Path);
for var = var_size
    b1=(var-1)*b1_max/20
    for trail= 1:num_trail
    [xi_a,Xi_b,xi_c] = HirPatterns(N,p_a,p_b,p_c,b1,b2);
    %Xi_b is the primitive matrix before the shape changes
    xi_a=reshape(xi_a,[N,p_a]);
    xi_b=reshape(Xi_b,[N,p_a*p_b]);
    xi_c=reshape(xi_c,[N,p_a*p_b*p_c]);
    
    %% interaction between layers
    W2= weight(xi_b);
    W1= weight(xi_c);
    % feedforward= c_ff*feedforward((xi_b+1)/2,(xi_c+1)/2);
    % pfeedback = c_fb*p_feedback((xi_b+1)/2,(xi_c+1)/2);
    feedforward= Feedforward(Xi_b,xi_c);
    pfeedback =feedforward';%(xi_b),(xi_c+1)/2
    nfeedback=-b1;
    % feedforward=b1^3*(p_c-1);
    % pfeedback = b1^3*(p_c-1);
    clear Xi_b
    
    %% inputs of each layers with time
    time_in=7;
    Times=25;dt=0.01;
    X_00=33;
    [x0]=deal(zeros(N,1));
    % x0(1:1*N,1)=xi_c(1:1*N,X_00);
    % x0(0.9*N:1*N,1)=xi_c(0.9*N:1*N,X_00+2*p_c);
    [I_ext,I_ext2,I_pfb,I_nfb]=deal(zeros(1,Times/dt));
    I_ext(1:Times/dt)=1;
    I_pfb((time_in-2)/dt:(time_in+3)/dt)=1;
    I_nfb((time_in+3)/dt:(time_in+8)/dt)=1;
    % n_x2=randperm(length(x1),0.3*N);x2(n_x2,1)=-x1(n_x2,1);
    % x2=x0;
   
    %% each trail

    x0=xi_c(:,X_00);
    n1_x0=randperm(length(x0),fix(lambda1*N));
    x0(n1_x0,1)=xi_c(n1_x0,X_00+2);
    n_x0=randperm(length(x0),fix(lambda2*N));
    x0(n_x0,1)=xi_c(n_x0,X_00+1*p_c);
    
    [h1,h2,x1,x2]=deal(zeros(N,1));
    [h11,x11]=deal(zeros(N,1));
    for i=1:Times/dt
        dh1=1/tau*(1*W1*(x1)-h1/B+...
            a_ext*x0*I_ext(i)+...
            1*pfeedback*(x2)*I_pfb(i)+...
            10*nfeedback*(x2)*I_nfb(i)+...
            0.1*(rand(N,1)-0.5))*dt;
        h1=h1+dh1;
        x1=0.5*((2/pi)*atan(8*pi*h1)+1);
        
        dh11=1/tau*(1*W1*(x11)-h11/B+...
            a_ext*x0*I_ext(i)+...
            0.1*(rand(N,1)-0.5))*dt;
        h11=h11+dh11;
        x11=0.5*((2/pi)*atan(8*pi*h11)+1);
        
        dh2=1/tau*(2*W2*(x2)-h2/B+...
            2*feedforward*(x1)+...
            0.1*x0*I_ext(i))*dt;
        h2=h2+dh2;
        x2=0.5*((2/pi)*atan(8*pi*h2)+1); 
    end
    m11(var,trail)=1/N*(sign(x11*2-1)'*sign(xi_c(:,X_00)*2-1));
    m1(var,trail)=1/N*(sign(x1*2-1)'*sign(xi_c(:,X_00)*2-1));
    m2(var,trail)=1/N*(sign(x2*2-1)'*sign(xi_b(:,ceil(X_00/p_c))*2-1));
    m_improv(var,trail)=(m1(var,trail)-m11(var,trail))/m11(var,trail);
    end
end
save([Path,'/','m11.mat'],'m11');
save([Path,'/','m1.mat'],'m1');
save([Path,'/','m2.mat'],'m2');
save([Path,'/','m_imp.mat'],'m_improv');
