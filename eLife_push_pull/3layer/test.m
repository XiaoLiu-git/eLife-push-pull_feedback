clear all
close all
a_ext=0.1;
N=2000;
p_a=2;
p_b=2;
p_c=30;
b1=0.1;
b2=0.1;
tau=5;
B=1;
n_trail = 30;
[X1_p_mean,X1_n_mean,X1_1_mean]=deal(zeros(1,n_trail));
for trail = 1:n_trail
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
    time_in3=7;
    time_in2=7;
    Time_total=25;
    dt=0.01;
    X_00=1;
    
    lambda1=0.4;
    lambda2=0.1;
    
    %% each trail
    [m1_push,m2_push,m1_pull,m2_pull,m1_2,m2_2,m1_1]=deal(zeros(1,Time_total/dt));
    x0=xi_c(:,X_00);
    %% generate noise
    n1_x0=randperm(length(x0),fix(lambda1*N));
    x0(n1_x0,1)=xi_c(n1_x0,X_00+2);
    n_x0=randperm(length(x0),fix(lambda2*N));
    x0(n_x0,1)=xi_c(n_x0,X_00+1*p_c);
   
    [I2_pfb,I2_nfb]=deal(zeros(1,Time_total/dt));
    [I1_ext]=deal(ones(1,Time_total/dt));
    I1_ext(1:time_in3/dt)=1;
    I_ext=ones(1,Time_total/dt);
%     [I_ext]=deal(zeros(1,Time_total/dt));
%     I_ext(1:time_in2/dt)=1;
%     
    I2_pfb((time_in2-2)/dt:(time_in2+3)/dt)=1;
    I2_nfb((time_in2+3)/dt:(time_in2+8)/dt)=1;

    
    [h1_pull,h2_pull,x1_pull,x2_pull]=deal(zeros(N,1));
    [h1_push,h2_push,x1_push,x2_push]=deal(zeros(N,1));
    [h1_2,h2_2,x1_2,x2_2]=deal(zeros(N,1));
    [h1_1,x1_1]=deal(zeros(N,1));
    
    for i=1:Time_total/dt
        
        %% 1 layer
        dh1_1=1/tau*(1*W1*(x1_1)-h1_1/B+...
            1*x0*I_ext(i)+...
            0.1*(rand(N,1)-0.5))*dt;
        h1_1=h1_1+dh1_1;
        x1_1=0.5*((2/pi)*atan(8*pi*h1_1)+1);
                      
        m1_1(i)=1/N*(sign(x1_1*2-1)'*sign(xi_c(:,X_00)*2-1));
    end
    
      for  i=1:Time_total/dt
        %% only push
        dh1_push=1/tau*(1*W1*(x1_push)-h1_push/B+...
            1*x0*I_ext(i)+...
            1*pfeedback12*(x2_push)*I2_pfb(i)+...
            0.1*(rand(N,1)-0.5))*dt; %
        h1_push=h1_push+dh1_push;
        x1_push=0.5*((2/pi)*atan(8*pi*h1_push)+1);

        dh2_push=1/tau*(2*W2*(x2_push)-h2_push/B+...
            6*feedforward21*(x1_push)+...
            a_ext*x0*I1_ext(i))*dt;%+0.01*(rand(N,1)-0.5)
        h2_push=h2_push+dh2_push;
        x2_push=0.5*((2/pi)*atan(8*pi*h2_push)+1);

        m1_push(i)=1/N*(sign(x1_push*2-1)'*sign(xi_c(:,ceil(X_00))*2-1));
        m2_push(i)=1/N*(sign(x2_push*2-1)'*sign(xi_b(:,ceil(X_00/p_c))*2-1));
      end
      
       for  i=1:Time_total/dt
        %% only pull
        dh1_pull=1/tau*(1*W1*(x1_pull)-h1_pull/B+...
            1*x0*I_ext(i)+...
            10*nfeedback12*(x2_pull)*I2_nfb(i)+...
            0.1*(rand(N,1)-0.5))*dt; %
        h1_pull=h1_pull+dh1_pull;
        x1_pull=0.5*((2/pi)*atan(8*pi*h1_pull)+1);
        
        dh2_pull=1/tau*(2*W2*(x2_pull)-h2_pull/B+...
            6*feedforward21*(x1_pull)+...
            a_ext*x0*I1_ext(i))*dt;%+0.01*(rand(N,1)-0.5)
        h2_pull=h2_pull+dh2_pull;
        x2_pull=0.5*((2/pi)*atan(8*pi*h2_pull)+1);

        m1_pull(i)=1/N*(sign(x1_pull*2-1)'*sign(xi_c(:,ceil(X_00))*2-1));
        m2_pull(i)=1/N*(sign(x2_pull*2-1)'*sign(xi_b(:,ceil(X_00/p_c))*2-1));
      end
       for  i=1:Time_total/dt
        %%  push and pull
        dh1_2=1/tau*(1*W1*(x1_2)-h1_2/B+...
            1*x0*I_ext(i)+...
            1*pfeedback12*(x2_2)*I2_pfb(i)+...
            10*nfeedback12*(x2_2)*I2_nfb(i)+...
            0.1*(rand(N,1)-0.5))*dt; %
        h1_2=h1_2+dh1_2;
        x1_2=0.5*((2/pi)*atan(8*pi*h1_2)+1);
        
        dh2_2=1/tau*(2*W2*(x2_2)-h2_2/B+...
            6*feedforward21*(x1_2)+...
            a_ext*x0*I1_ext(i))*dt;%+0.01*(rand(N,1)-0.5)
        h2_2=h2_2+dh2_2;
        x2_2=0.5*((2/pi)*atan(8*pi*h2_2)+1);

        m1_2(i)=1/N*(sign(x1_2*2-1)'*sign(xi_c(:,ceil(X_00))*2-1));
        m2_2(i)=1/N*(sign(x2_2*2-1)'*sign(xi_b(:,ceil(X_00/p_c))*2-1));
      end
 X1_p_mean(trail)=(m1_push(end)-m1_1(end))/m1_1(end);
 X1_n_mean(trail)=(m1_pull(end)-m1_1(end))/m1_1(end);
 X1_1_mean(trail)=(m1_2(end)-m1_1(end))/m1_1(end);   

    figure();
    Overlaps=plot((dt:dt:Time_total)/tau,m1_1,':b',...
        (dt:dt:Time_total)/tau,m1_2,':r',...
        (dt:dt:Time_total)/tau,m2_2,':y',...
        (dt:dt:Time_total)/tau,m1_pull,'r',...
        (dt:dt:Time_total)/tau,m1_push,'y');
    set(Overlaps,'linewidth',3)
    xlabel('Time(\tau)')
    ylabel('Retrieval Accuracy')
    legend('m^1 without fb',' m^1_2 ',' m^2_2 ',' m^1 only pull',' m^1 only push ')
    figure_FontSize=20;
    set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
    set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
    set(findobj('FontSize',10),'FontSize',figure_FontSize);
    %     saveas(gcf,[Path,'/','MisingCurve',num2str(trial),'.eps'],'psc2');
  
end

