dt=0.01;
for b1=0.25;
    b2=0.1;
    P_c=80;
        for P_b=20;
        mu1=b1^3*(P_c-1);
        mu11c=b1^3*(P_c-1)-b1;
        mut=b1^3*b2^3*P_c*(P_b-1);
        S_c=b1^4*(P_c-1)*(1-b1^2);
        S_ct=b1^4*b2^4*P_c*(P_b-1)*(1-b1^2)*(1-b2^2);
        x=-3:dt:3;
        y1=(1+b1)/2*exp(-(x-mu1).^2/(2*S_c))+(1-b1)/2*exp(-(x+mu1).^2/(2*S_c));
        y1=y1/sqrt(S_c*2*pi);
        y2=(1+b1)/2*exp(-(x-mu1+b1).^2/(2*S_c))+(1-b1)/2*exp(-(x+mu1-b1).^2/(2*S_c));
        y2=y2/sqrt(S_c*2*pi);
        yt=(1+b1*b2)/2*exp(-(x-mut).^2/(2*S_ct))+(1-b1*b2)/2*exp(-(x+mut).^2/(2*S_ct));
        yt=yt/sqrt(S_ct*2*pi);
        figure()
        plot_c_i=plot(x,y1,x,y2,-1*ones(1,401),linspace(0,0.6,401))
        title(['b1=',num2str(b1),'P_b=',num2str(P_b)])
        set(plot_c_i,'linewidth',3)
        legend('c_i','c_i^*')
        figure_FontSize=20;
        set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
        set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
        set(findobj('FontSize',10),'FontSize',figure_FontSize);
        hold on
%         figure()

        plot_c_i=plot(x,y1,x,yt,-1*ones(1,401),linspace(0,1.2,401))
        title(['b1=',num2str(b1),'P_b=',num2str(P_b)])
        set(plot_c_i,'linewidth',3)
%         legend('c_i','c_i^t')
        figure_FontSize=20;
        set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
        set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
        set(findobj('FontSize',10),'FontSize',figure_FontSize);
    end
end
