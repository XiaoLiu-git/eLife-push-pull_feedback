clear all
close all
a_ext=0.1;
N=1000;
p_a=2;
p_b=4;
p_c=25;
b1=0.2;
b2=0.2;
tau=5;
B=1;
for trail = 1:10
    [Xi_a,Xi_b,xi_c] = HirPatterns(N,p_a,p_b,p_c,b1,b2);
    %Xi_b is the primitive matrix before the shape changes
    xi_a=reshape(Xi_a,[N,p_a]);
    xi_b=reshape(Xi_b,[N,p_a*p_b]);
    xi_c=reshape(xi_c,[N,p_a*p_b*p_c]);
    
    %% interaction between layers
    W3 = weight(xi_a);
    W2 = weight(xi_b);
    W1 = weight(xi_c);
    feedforward21 = Feedforward(Xi_b,xi_c);
    feedforward32 = Feedforward(Xi_a,xi_b);
    % feedback
    pfeedback12 = feedforward21';
    nfeedback12 = -b1;
    pfeedback23 = feedforward32';
    nfeedback23 = -b2;
    clear Xi_b Xi_a
    
    %% inputs of each layers with time
    time_in3=3;
    time_in2=11;
    Time_total=30;
    dt=0.01;
    X_00=1;
    
    lambda1=0.15;
    lambda2=0.15;
    
    %% each trail
    [m1_3,m2_3,m3_3,m1_2,m2_2,m1_1]=deal(zeros(1,Time_total/dt));
    [X1_3_mean,X1_2_mean,X1_1_mean]=deal(zeros(1,Time_total/dt));
    x0=xi_c(:,X_00);
    %% generate noise
    n1_x0=randperm(length(x0),fix(lambda1*N));
    x0(n1_x0,1)=xi_c(n1_x0,X_00+2);
    n_x0=randperm(length(x0),fix(lambda2*N));
    x0(n_x0,1)=xi_c(n_x0,X_00+1*p_c);
     
    [I2_pfb,I2_nfb,I2_ext,I1_ext]=deal(zeros(1,Time_total/dt));
    [I3_pfb,I3_nfb,I3_ext]=deal(zeros(1,Time_total/dt));
    I1_ext(1:Time_total/dt)=1;

    
    I3_pfb((time_in3-2)/dt:(time_in3+3)/dt)=1;
    I3_nfb((time_in3+3)/dt:(time_in3+8)/dt)=1;
    I3_ext(1:time_in3/dt)=1;
    
    I2_pfb((time_in2-2)/dt:(time_in2+3)/dt)=1;
    I2_nfb((time_in2+3)/dt:(time_in2+8)/dt)=1;
    I2_ext(1:time_in2/dt)=1;   

    for  a_re2=[0.5,1,1.5,2]
    a_ext=0.1;
    
    [h1_3,h2_3,h3_3,x1_3,x2_3,x3_3]=deal(zeros(N,1));
    [h1_2,h2_2,h3_2,x1_2,x2_2,x3_2]=deal(zeros(N,1));
    [h1_1,x1_1]=deal(zeros(N,1));
    
    for i=1:Time_total/dt
        
        %% 1 layer no push pull
        dh1_1=1/tau*(1*W1*(x1_1)-h1_1/B+...
            1*x0*I1_ext(i))*dt;%+...
%             0.1*(rand(N,1)-0.5)
        h1_1=h1_1+dh1_1;
        x1_1=0.5*((2/pi)*atan(8*pi*h1_1)+1);
                      
        m1_1(i)=1/N*(sign(x1_1*2-1)'*sign(xi_c(:,X_00)*2-1));
        X1_1_mean(i)=mean(x1_1,1);
    end
    
      for  i=1:Time_total/dt
        %% 2 layers push-pull 1&2 
        dh1_2=1/tau*(1*W1*(x1_2)-h1_2/B+...
            1*x0*I1_ext(i)+...
            a_re2*pfeedback12*(x2_2)*I2_pfb(i)+...
            5*nfeedback12*(x2_2)*I2_nfb(i))*dt; %+...
%             0.1*(rand(N,1)-0.5)
        h1_2=h1_2+dh1_2;
        x1_2=0.5*((2/pi)*atan(8*pi*h1_2)+1);
        
        dh2_2=1/tau*(2*W2*(x2_2)-h2_2/B+...
            6*feedforward21*(x1_2))*dt;%+0.01*(rand(N,1)-0.5)
        h2_2=h2_2+dh2_2;
        x2_2=0.5*((2/pi)*atan(8*pi*h2_2)+1);
        
        dh3_2=1/tau*(3*W3*(x3_2)-h3_2/B+...
            6*feedforward32*(x3_2)+...
            a_ext*x0*I3_ext(i))*dt;%+0.01*(rand(N,1)-0.5)
        h3_2=h3_2+dh3_2;
        x3_2=0.5*((2/pi)*atan(8*pi*h3_2)+1);

        m1_2(i)=1/N*(sign(x1_2*2-1)'*sign(xi_c(:,X_00)*2-1));
        m2_2(i)=1/N*(sign(x2_2*2-1)'*sign(xi_b(:,ceil(X_00/p_c))*2-1));
        m3_2(i)=1/N*(sign(x3_2*2-1)'*sign(xi_a(:,ceil(X_00/p_c/p_b))*2-1));
        X1_2_mean(i)=mean(x1_2,1);
      end
      
      for i=1:Time_total/dt
        %% 3 layers push pull 1&2 2&3
        dh1_3=1/tau*(1*W1*(x1_3)-h1_3/B+...
            1*x0*I1_ext(i)+...
            a_re2*pfeedback12*(x2_3)*(I2_pfb(i)+I3_pfb(i))+...
            5*nfeedback12*(x2_3)*(I2_nfb(i)+I3_nfb(i)))*dt;%+...
%             0.1*(rand(N,1)-0.5)
        h1_3=h1_3+dh1_3;
        x1_3=0.5*((2/pi)*atan(8*pi*h1_3)+1);
        
        dh2_3=1/tau*(2*W2*(x2_3)-h2_3/B+...
            6*feedforward21*(x1_3)+...
            2*pfeedback23*(x3_3)*(I2_pfb(i)+I3_pfb(i))+...
            7*nfeedback23*(x3_3)*(I2_nfb(i)+I3_nfb(i)))*dt;%+0.01*(rand(N,1)-0.5)
        h2_3=h2_3+dh2_3;
        x2_3=0.5*((2/pi)*atan(8*pi*h2_3)+1);
        
        dh3_3=1/tau*(3*W3*(x3_3)-h3_3/B+...
            6*feedforward32*(x2_3)+...
            a_ext*x0*I3_ext(i))*dt;%+0.01*(rand(N,1)-0.5)
        h3_3=h3_3+dh3_3;
        x3_3=0.5*((2/pi)*atan(8*pi*h3_3)+1);

        m1_3(i)=1/N*(sign(x1_3*2-1)'*sign(xi_c(:,X_00)*2-1));
        m2_3(i)=1/N*(sign(x2_3*2-1)'*sign(xi_b(:,ceil(X_00/p_c))*2-1));
        m3_3(i)=1/N*(sign(x3_3*2-1)'*sign(xi_a(:,ceil(X_00/p_c/p_b))*2-1));
        X1_3_mean(i)=mean(x1_3,1);
    end

    figure();
    subplot(1,2,1)
    Overlaps=plot((dt:dt:Time_total)/tau,m1_1,':b',...
        (dt:dt:Time_total)/tau,m1_2,':r',...
        (dt:dt:Time_total)/tau,m2_2,':y',...
        (dt:dt:Time_total)/tau,m3_2,':c',...
        (dt:dt:Time_total)/tau,m1_3,'r',...
        (dt:dt:Time_total)/tau,m2_3,'y',...
        (dt:dt:Time_total)/tau,m3_3,'c');
    set(Overlaps,'linewidth',3)
    xlabel('Time(\tau)')
    ylabel('Retrieval Accuracy')
    legend('m^1 without fb',' m^1_2 ',' m^2_2 ','m^3_2',' m^1_3',' m^2_3 ',' m^3_3')
    figure_FontSize=20;
    set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
    set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
    set(findobj('FontSize',10),'FontSize',figure_FontSize);
    title(['a_{pull}=',num2str(a_re2)])
    %     saveas(gcf,[Path,'/','MisingCurve',num2str(trial),'.eps'],'psc2');
    subplot(1,2,2)
    Overlaps=plot((dt:dt:Time_total)/tau,X1_1_mean,'b',...
        (dt:dt:Time_total)/tau,X1_2_mean,'r',...
        (dt:dt:Time_total)/tau,X1_3_mean,'c');
    set(Overlaps,'linewidth',3)
    ylabel('Activity ⟨x⟩')
    xlabel('Time(\tau)')
    legend('<x^1> without fb',' <x^1_2> ',' <x^1_3> ')
    figure_FontSize=20;
    set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
    set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
    set(findobj('FontSize',10),'FontSize',figure_FontSize);
    title(['a_{pull}=',num2str(a_re2)])
    %     saveas(gcf,[Path,'/','MisingCurve',num2str(trial),'.eps'],'psc2');
    end
        fprintf(1,'finished...%d!\n',trail)
end

