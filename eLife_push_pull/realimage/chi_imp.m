Path=['./results/',datestr(now,6),'/realimage'];
mkdir(Path);
[raw_instance,Pattern_par] = LoadData(1);%load raw_instance(0,1),Pattern_par(0,1)
[num_gra,num_par,num_chi]=deal(1,2,9);
[N,num_pat]=size(Pattern_par);
Pattern_gra=Gen_highlayer(Pattern_par,num_chi);
wt_par = Hebb_weight( Pattern_par );
wt_gra = Hebb_weight( Pattern_gra );
b1=0.4;
b2=0.17;
tau=5;
B=1;
feedforward= Feedforward1(Pattern_gra,Pattern_par,num_gra,num_par);
pfeedback =feedforward';%(xi_b),(xi_c+1)/2
nfeedback=-b1;
Fun=0;
%% inputs of each layers with time
Times=20;
dt=0.01;
time_in=7;
T=0.01; %for sampling
[M1,M2,M11]=deal(zeros(1,1800));
for Num_chi=1:1800;
    [I_ext,I_ext2,I_pfb,I_nfb]=deal(zeros(1,Times/dt));
    I_ext(1:Times/dt)=1;
    I_pfb((time_in-2)/dt:(time_in+5)/dt)=1;
    I_nfb((time_in+5)/dt:(time_in+12)/dt)=1;
    
    %% each trail
    [h1,h2,x1,x2]=deal(zeros(N,1));
    [h11,x11]=deal(zeros(N,1));
    Tn=ceil(T/dt);
    x0=raw_instance(:,Num_chi);
    for ti=1:Times/T
        for i=1:Tn
            % push-pull feedback
            dh1=1/tau*(1*wt_par*(x1)-h1/B+...
                6*x0*I_ext((ti-1)*Tn+i)+...
                2*pfeedback*(x2)*I_pfb((ti-1)*Tn+i)+...
                1.5*nfeedback*(x2)*I_nfb((ti-1)*Tn+i))*dt;%+...1*(rand(N,1)-0.5)
            h1=h1+dh1;
            x1=0.5*((2/pi)*atan(8*pi*h1)+1);
            % Layer 2
            dh2=1/tau*(2*wt_gra*(x2)-h2/B+...
                1*x0*I_ext((ti-1)*Tn+i)+...
                1*feedforward*(x0))*dt;
            h2=h2+dh2;
            x2=0.5*((2/pi)*atan(8*pi*h2)+1);
            % Without feedback
            dh11=1/tau*(1*wt_par*(x11)-h11/B+...
                6*x0*I_ext((ti-1)*Tn+i))*dt;%+...           1*(rand(N,1)-0.5)
            h11=h11+dh11;
            x11=0.5*((2/pi)*atan(8*pi*h11)+1);
            
        end
    end
    %% save overlap
    % Overlap of Neural activity
    M11(Num_chi)=noverlap(x11,Pattern_par(:,ceil(Num_chi/100)),Fun);
    M1(Num_chi)=noverlap(x1,Pattern_par(:,ceil(Num_chi/100)),Fun);
    M2(Num_chi)=noverlap(x2,Pattern_gra(:,ceil(Num_chi/num_chi/100)),Fun);
    save([Path,'/','m_11.mat'],'M11');
    save([Path,'/','m_1.mat'],'M1');
    save([Path,'/','m_2.mat'],'M2');
end
