clear all
%% normalize pattern
MM=13;
load('anm43_act.mat')%fc2s  the shape is 4096, 4*3*200 supclass*subclass*examples
raw_instance=log(1+double(fc2s'));
Mean_fc2s=ones(4096,1)*mean(raw_instance,1);
Std_fc2s=ones(4096,1)*std(raw_instance,0,1);
Std_fc2s(Std_fc2s==0)=1;
raw_instance=(raw_instance-Mean_fc2s)./(2*Std_fc2s)+0.5;%
raw_instance(raw_instance<0)=0;
raw_instance=sign(raw_instance-mean(mean(raw_instance)))*0.5+0.5;
x=sign(raw_instance-0.5);
x(:,401:800)=[];
m=1/4096*(x'*x);
n_child=100;
figure;
imagesc(m);caxis([-0.1 1 ]);colorbar;set(gca,'linewidth',2);
M=size(m, 1)+1;N=size(m, 2)+1;
hold on;
[xt, yt] = meshgrid(round(linspace(1,M,MM)), ...
    round(linspace(1, N, MM)));%生成数据点矩阵
mesh(yt-0.5, xt-0.5, zeros(size(xt)), 'FaceColor', ...
    'None', 'LineWidth', 1, ...
    'EdgeColor', 'w');%绘制三维网格图
figure_FontSize=20;
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
set(findobj('FontSize',10),'FontSize',figure_FontSize);
title(' M_{1} ')
%% Select x
new_inst=raw_instance;
% for index=0:11
%     [max_overlap,loc]=sort(mean(m(1+index*100:(1+index)*100,1+index*100:(1+index)*100)));%,'descend'
%     new_inst(:,1+index*100:(1+index)*100)=raw_instance(:,index*100+sort(loc(1:100)));
% end
new_inst(:,401:800)=[];
x=sign(new_inst-0.5);
% x(:,401:800)=[];
m=1/4096*(x'*x);
figure;
imagesc(m);caxis([-0.1 1 ]);colorbar;set(gca,'linewidth',2);
M=size(m, 1)+1;N=size(m, 2)+1;
hold on;
[xt, yt] = meshgrid(round(linspace(1,M,MM)), ...
    round(linspace(1, N, MM)));%生成数据点矩阵
mesh(yt-0.5, xt-0.5, zeros(size(xt)), 'FaceColor', ...
    'None', 'LineWidth', 1, ...
    'EdgeColor', 'w');%绘制三维网格图
figure_FontSize=20;
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
set(findobj('FontSize',10),'FontSize',figure_FontSize);
title(' M_{1} ')
%% generate subclass-patterns
subclass_pattern=zeros(4096,12);
for index=0:11
    subclass_pattern(:,index+1)=mean(new_inst(:,index*n_child+1:(index+1)*n_child),2);
end
subclass_pattern=sign(subclass_pattern-mean(mean(subclass_pattern)))/2+0.5;
I= subclass_pattern==0.5;%%找到pattern_parent中所有等于0.5的元素
subclass_pattern(I)=0;
x=sign(subclass_pattern-0.5);
m=1/4096*(x'*x);
figure;
imagesc(m);caxis([-0.1 1 ]);colorbar;set(gca,'linewidth',2);
M=size(m, 1)+1;N=size(m, 2)+1;
hold on;
[xt, yt] = meshgrid(round(linspace(1,M,4)), ...
    round(linspace(1, N, 4)));%生成数据点矩阵
mesh(yt-0.5, xt-0.5, zeros(size(xt)), 'FaceColor', ...
    'None', 'LineWidth', 1, ...
    'EdgeColor', 'w');%绘制三维网格图
figure_FontSize=20;
set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');
set(get(gca,'YLabel'),'FontSize',figure_FontSize,'Vertical','middle');
set(findobj('FontSize',10),'FontSize',figure_FontSize);
title(' M_{1} ')
%% generate supclass-patterns
supclass_pattern=zeros(4096,3);
for index=0:2
    supclass_pattern(:,index+1)=mean(subclass_pattern(:,index*4+1:index*4+4),2);
end
supclass_pattern=sign(supclass_pattern-mean(mean(supclass_pattern)))/2+0.5;
I= supclass_pattern==0.5;%%找到pattern_parent中所有等于0.5的元素
supclass_pattern(I)=0;
x=sign(supclass_pattern-0.5);
m=1/4096*(x'*x);
figure;
imagesc(m);caxis([-0.1 1 ]);colorbar;set(gca,'linewidth',2);