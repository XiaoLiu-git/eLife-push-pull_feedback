clear all
a_ext=1;
num_trail=50;
var_size=1:11;
N=1000;
p_a=2;
p_b=4;
p_c=15;
b1=0.15;
b2=0.1;
tau=5;
B=1;
lambda1_max=0.4;
lambda2=0.1;
[m_improv,m_p_improv,m_n_improv]=deal(zeros(max(var_size),num_trail));
Path=['./results/',datestr(now,6),'/push_pull_lmd1'];
mkdir(Path);
for var = var_size
    lambda1=(var-1)*lambda1_max/10
    for trail= 1:num_trail
        [xi_a,Xi_b,xi_c] = HirPatterns(N,p_a,p_b,p_c,b1,b2);
        %Xi_b is the primitive matrix before the shape changes
        xi_a=reshape(xi_a,[N,p_a]);
        xi_b=reshape(Xi_b,[N,p_a*p_b]);
        xi_c=reshape(xi_c,[N,p_a*p_b*p_c]);
        
        %% interaction between layers
        W2= weight(xi_b);
        W1= weight(xi_c);
        feedforward= Feedforward(Xi_b,xi_c);
        pfeedback =feedforward';%(xi_b),(xi_c+1)/2
        nfeedback=-b1;
        clear Xi_b
        
        %% inputs of each layers with time
        time_in=7;
        Times=25;dt=0.01;
        X_00=33;
        [x0]=deal(zeros(N,1));
        [I_ext,I_ext2,I_pfb,I_nfb]=deal(zeros(1,Times/dt));
        I_ext(1:Times/dt)=1;
        I_pfb((time_in-2)/dt:(time_in+3)/dt)=1;
        I_nfb((time_in+3)/dt:(time_in+8)/dt)=1;
        
        %% each trail
        x0=xi_c(:,X_00);
        n1_x0=randperm(length(x0),fix(lambda1*N));
        x0(n1_x0,1)=xi_c(n1_x0,X_00+2);
        n_x0=randperm(length(x0),fix(lambda2*N));
        x0(n_x0,1)=xi_c(n_x0,X_00+1*p_c);
        
        [h1,h2,x1,x2]=deal(zeros(N,1));
        [h1_p,h2_p,x1_p,x2_p]=deal(zeros(N,1));
        [h1_n,h2_n,x1_n,x2_n]=deal(zeros(N,1));
        [h11,x11]=deal(zeros(N,1));
        for i=1:Times/dt
            %% without feedback
            dh11=1/tau*(1*W1*(x11)-h11/B+...
                a_ext*x0*I_ext(i)+...
                0.1*(rand(N,1)-0.5))*dt;
            h11=h11+dh11;
            x11=0.5*((2/pi)*atan(8*pi*h11)+1);
            
            %% with push feedback
            dh1_p=1/tau*(1*W1*(x1_p)-h1_p/B+...
                a_ext*x0*I_ext(i)+...
                1*pfeedback*(x2_p)*I_pfb(i)+...
                0.1*(rand(N,1)-0.5))*dt;
            h1_p=h1_p+dh1_p;
            x1_p=0.5*((2/pi)*atan(8*pi*h1_p)+1);
            
            dh2_p=1/tau*(2*W2*(x2_p)-h2_p/B+...
                2*feedforward*(x1_p)+...
                0.1*x0*I_ext(i))*dt;
            h2_p=h2_p+dh2_p;
            x2_p=0.5*((2/pi)*atan(8*pi*h2_p)+1);
                       
            %% with pull feedback
            dh1_n=1/tau*(1*W1*(x1_n)-h1_n/B+...
                a_ext*x0*I_ext(i)+...
                10*nfeedback*(x2_n)*I_nfb(i)+...
                0.1*(rand(N,1)-0.5))*dt;
            h1_n=h1_n+dh1_n;
            x1_n=0.5*((2/pi)*atan(8*pi*h1_n)+1);
            
            dh2_n=1/tau*(2*W2*(x2_n)-h2_n/B+...
                2*feedforward*(x1_n)+...
                0.1*x0*I_ext(i))*dt;
            h2_n=h2_n+dh2_n;
            x2_n=0.5*((2/pi)*atan(8*pi*h2_n)+1);
            
           %% with push-pull feedback
            dh1=1/tau*(1*W1*(x1)-h1/B+...
                a_ext*x0*I_ext(i)+... 
                1*pfeedback*(x2_p)*I_pfb(i)+...
                10*nfeedback*(x2)*I_nfb(i)+...
                0.1*(rand(N,1)-0.5))*dt;
            h1=h1+dh1;
            x1=0.5*((2/pi)*atan(8*pi*h1)+1);
            
            dh2=1/tau*(2*W2*(x2)-h2/B+...
                2*feedforward*(x1)+...
                0.1*x0*I_ext(i))*dt;
            h2=h2+dh2;
            x2=0.5*((2/pi)*atan(8*pi*h2)+1);
        end
        m11=1/N*(sign(x11*2-1)'*sign(xi_c(:,X_00)*2-1));
        
        m1=1/N*(sign(x1*2-1)'*sign(xi_c(:,X_00)*2-1));
        m_improv(var,trail)=(m1-m11)/m11;
        
        m1_p=1/N*(sign(x1_p*2-1)'*sign(xi_c(:,X_00)*2-1));
        m_p_improv(var,trail)=(m1_p-m11)/m11;
        
        m1_n=1/N*(sign(x1_n*2-1)'*sign(xi_c(:,X_00)*2-1));
        m_n_improv(var,trail)=(m1_n-m11)/m11;
    end
end
save([Path,'/','m_improv.mat'],'m_improv');
save([Path,'/','m_p_improv.mat'],'m_p_improv');
save([Path,'/','m_n_improv.mat'],'m_n_improv');
