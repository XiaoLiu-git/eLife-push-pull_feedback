clear all
Path=['./results/',datestr(now,6),'/dogs'];
mkdir(Path);
[raw_instance,child_pattern,parent_pattern] = LoadData(2);%load raw_instance(0,1),Pattern_par(0,1)
[num_gra,num_par,num_chi]=deal(1,3,4);
[N,num_pat]=size(child_pattern);
wt_chi = Hebb_weight( child_pattern );
wt_par = Hebb_weight( parent_pattern );
b1=0.4;
b2=0.17;
tau=5;
B=1;
feedforward= Feedforward1(parent_pattern,child_pattern,num_gra,num_par);
pfeedback =feedforward';%(xi_b),(xi_c+1)/2
nfeedback=-b1;
Fun=0;
%% inputs of each layers with time
Times=24;
dt=0.01;
time_in=7;
T=0.01; %for sampling
a_ext=3;
a_n=4;
[M1,M2,M11,M_ipv]=deal(zeros(1,400));
for Num_chi=801:1200;
    [I_ext,I_ext2,I_pfb,I_nfb]=deal(zeros(1,Times/dt));
    I_ext(5/dt:Times/dt)=1;
    I_ext2(5/dt:time_in/dt)=1;
    I_pfb((time_in-2+5)/dt:(time_in+5+5)/dt)=1;
    I_nfb((time_in+5+5)/dt:(time_in+12+5)/dt)=1;
    
    %% each trail
    [x0]=deal(zeros(N,1));
    [h1,h2,x1,x2]=deal(zeros(N,1));
    [h11,x11]=deal(zeros(N,1));
    [h1_push,h1_pull,x1_push,x1_pull]=deal(zeros(N,1));
    [n1,n11]=deal(zeros(1,Times/dt));
    [h2_push,h2_pull,x2_push,x2_pull]=deal(zeros(N,1));
    Tn=ceil(T/dt);
    x0=raw_instance(:,Num_chi);
    for ti=5/T:Times/T
        for i=1:Tn
            % push-pull feedback
            dh1=1/tau*(1*wt_chi*(x1)-h1/B+...
                a_ext*x0*I_ext((ti-1)*Tn+i)+...
                2*pfeedback*(x2)*I_pfb((ti-1)*Tn+i)+...
                a_n*nfeedback*(x2)*I_nfb((ti-1)*Tn+i))*dt;%+...1*(rand(N,1)-0.5)
            h1=h1+dh1;
            x1=0.5*((2/pi)*atan(8*pi*h1)+1);
            % Layer 2
            dh2=1/tau*(2*wt_par*(x2)-h2/B+...
                0.1*x0*I_ext2((ti-1)*Tn+i)+...
                3*feedforward*(x0))*dt;
            h2=h2+dh2;
            x2=0.5*((2/pi)*atan(8*pi*h2)+1);

            % Without feedback
            dh11=1/tau*(1*wt_chi*(x11)-h11/B+...
                a_ext*x0*I_ext((ti-1)*Tn+i))*dt;%+...           1*(rand(N,1)-0.5)
            h11=h11+dh11;
            x11=0.5*((2/pi)*atan(8*pi*h11)+1);
            % Overlap of Neural activity
            %             m11((ti-1)*Tn+i)=noverlap(x11,child_pattern(:,ceil(Num_chi/100)),Fun);
            %             m1((ti-1)*Tn+i)=noverlap(x1,child_pattern(:,ceil(Num_chi/100)),Fun);
            %             m2((ti-1)*Tn+i)=noverlap(x2,parent_pattern(:,ceil(Num_chi/num_chi/100)),Fun);
        end
    end
    %% save overlap
    M1(Num_chi) = noverlap(x1,child_pattern(:,ceil(Num_chi/100)),Fun);
    M11(Num_chi) = noverlap(x11,child_pattern(:,ceil(Num_chi/100)),Fun);
    M2(Num_chi)=noverlap(x2,parent_pattern(:,ceil(Num_chi/num_chi/100)),Fun);
    M_ipv(Num_chi)=(M1(Num_chi)-M11(Num_chi))/abs(M11(Num_chi));
    save([Path,'/','M11.mat'],'M11');
    save([Path,'/','M1.mat'],'M1');
    save([Path,'/','M2.mat'],'M2');
    save([Path,'/','M_imp.mat'],'M_ipv');
    
end


