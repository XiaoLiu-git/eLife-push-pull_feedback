clear all
close all
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
Times=25;
dt=0.01;
time_in=7;
T=0.01; %for sampling

for Num_chi=13:10:400;
    for a_ext=[3]
        for a_n=4
    m=zeros(num_chi,num_par*num_gra,100);
    [I_ext,I_ext2,I_pfb,I_nfb]=deal(zeros(1,Times/dt));
    I_ext(5/dt:Times/dt)=1;
    I_ext2(5/dt:time_in/dt)=1;
    I_pfb((time_in-2+5)/dt:(time_in+5+5)/dt)=1;
    I_nfb((time_in+5+5)/dt:(time_in+12+5)/dt)=1;
      
    %% each trail
    [x0]=deal(zeros(N,1));
    [m1,m2,m11]=deal(zeros(1,Times/dt));
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
            m11((ti-1)*Tn+i)=noverlap(x11,child_pattern(:,ceil(Num_chi/100)),Fun);
            m1((ti-1)*Tn+i)=noverlap(x1,child_pattern(:,ceil(Num_chi/100)),Fun);
            m2((ti-1)*Tn+i)=noverlap(x2,parent_pattern(:,ceil(Num_chi/num_chi/100)),Fun);
            n11((ti-1)*Tn+i)=sum(x11);
            n1((ti-1)*Tn+i)=sum(x1);
  
        end
m(:,:,ti) = m_overlapTi(x1,child_pattern,num_chi,0);    
    end
    %% save overlap
figure;
h=plot((dt:dt:Times)./tau-1,m11*100,'b-',...
    (dt:dt:Times)./tau-1,m1*100,'r-',...
    (dt: dt:Times)./tau-1,m2*100,'y-');
set(gca,'YGrid','on');
set(gca,'GridLineStyle','-');
set(get(gca,'YLabel'),'String','Retrieval Accuracy');
set(get(gca,'XLabel'),'String','Time(\tau)');
lh = legend('m^{1,1,1} without fb','m^{1,1,1} with fb','m^{1,1} ');
set(lh,'Location','BestOutside','Orientation','horizontal');
set(findobj('FontSize',10),'FontSize',15);
ylb_lim=get(gca,'YTick');
ylb=get(gca,'YTickLabel');%得到原本x轴的标注，是一列字符串不含百分号
n=length(ylb);%得到标注的个数，即长度
a='%';
per=string(repmat(a,n,1));%构造一个相同长度的%的列
new_ylb=join([string(ylb) per]);%把百分号加到原标注的后面，即两个列字符串拼起来
set(gca,'YTickLabel',new_ylb); %将新的标注设为当前x轴的标注
set(h,'linewidth',3);

% m_overlapTi(x1,child_pattern,4,1);

figure;
h=plot((dt:dt:Times)./tau-1,n11/4096,'b-',...
    (dt:dt:Times)./tau-1,n1/4096,'r-');
set(gca,'YGrid','on');
set(gca,'GridLineStyle','-');
set(get(gca,'YLabel'),'String','Averaged Activity <x>');
set(get(gca,'XLabel'),'String','Time(\tau)');
lh = legend('<x> without fb','<x> push-pull');
set(lh,'Location','BestOutside','Orientation','horizontal')
set(findobj('FontSize',10),'FontSize',15);
set(h,'linewidth',3)

figure;
h=plot((dt:dt:Times)./tau-1,m1*100,...
         (dt:dt:Times)./tau-1,reshape(mean(m(:,1,:)),[1,2500])*100,...
         (dt:dt:Times)./tau-1,reshape(mean(m(:,2,:)),[1,2500])*100);
set(gca,'YGrid','on');
set(gca,'GridLineStyle','-');
set(get(gca,'YLabel'),'String','Retrieval Accuracy');
set(get(gca,'XLabel'),'String','Time(\tau)');
lh = legend('m^{cat A}','<m>^{cat}','<m>^{dog}');
set(lh,'Location','BestOutside','Orientation','horizontal');
set(findobj('FontSize',10),'FontSize',15);
axis([-inf,inf,min(ylb_lim),max(ylb_lim)]);
set(gca,'YTickLabel',new_ylb); %将新的标注设为当前x轴的标注
set(h,'linewidth',3);
    end
    end
end

