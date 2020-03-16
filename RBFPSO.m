clc;
clear;
tic;
SamNum=48;                         %训练样本数
TargetSamNum=3;                   %测试样本数
InDim=1;                            %样本输入维数
UnitNum=2;                          %隐节点数
MaxEpoch=1200;                      %最大训练次数
num=2;%对应四个特征
%E0=0.2;                             %目标误差
gbesthistory=[];
% 根据目标函数获得样本输入输出（训练样本）
rand('state',sum(100*clock));
%NoiseVar=0.0005;
%Noise=NoiseVar*randn(1,SamNum);
load data
%归一化
data1=data';
data=mapminmax(data1,0,1);
data=data';

%建立训练集测试集
x_train=[data(1:48,1).';data(1:48,2).';data(1:48,3).';data(1:48,4).'];
x_test=[data(2:49,num).'];
y_train=[data(49:51,1).';data(49:51,2).';data(49:51,3).';data(49:51,4).'];
y_test=[data(50:52,num).'];
SamIn=x_train;
SamOut=x_test;
%测试样本
TargetIn=y_train;
TargetOut=y_test;


%粒子群算法中的两个参数
c1 = 1.49445;
c2 = 1.49445;
popcount=10;   %粒子数
poplength=6;  %粒子维数
Wstart=0.9;%初始惯性权值
Wend=0.2;%迭代次数最大时惯性权值
%个体和速度最大最小值
Vmax=1;
Vmin=-1;
popmax=4;
popmin=-4;
%粒子位置速度和最优值初始化

for i=1:popcount
    pop(i,:)=rand(1,9);%初始化粒子位置
    V(i,:)=rand(1,9);%初始化粒子速度
    %计算粒子适应度值
    Center=pop(i,1:3);
    SP=pop(i,4:6); 
    W=pop(i,7:9);
    Distance=dist(Center',SamIn);
    SPMat=repmat(SP',1,SamNum);%repmat具体作用
    UnitOut=radbas(Distance./SPMat);%径向基函数
    NetOut=W*UnitOut;%网络输出
    Error=SamOut-NetOut;%网络误差
    %SSE=sumsqr(Error);
    %fitness(i)=SSE;
    RMSE=sqrt(sumsqr(Error)/SamNum);
    fitness(i)=RMSE;
    %fitness(i)=fun(pop(i,:));
end
%适应度函数（适应度值为RBF网络均方差）


[bestfitness bestindex]=min(fitness);
gbest=pop(bestindex,:);%全局最优值
pbest=pop;%个体最优值
pbestfitness=fitness;%个体最优适应度值
gbestfitness=bestfitness;%全局最优适应度值
%迭代寻优
for i=1:MaxEpoch
   Vmax=1.00014^(-i);
   Vmin=-1.00014^(-i);
    for j=1:popcount
       % if (fitness(j)<gbestfitness|fitness==gbestfitness)
           % S(j)=0;
        %end
        %S(j)=1-(fitness(j)/100)^2;
       % GW(j)=Wstart-S(j)*(Wstart-Wend);
       % GW(j)=Wend+(GW(j)-Wend)*(MaxEpoch-i)/MaxEpoch;
        GW=Wstart-(Wstart-Wend)*i/MaxEpoch;
        %速度更新(第一种方法精度最高)
        V(j,:) = 1.000009^(-i)*(gbestfitness/fitness(j)+2)*rand*V(j,:) + c1*rand*(pbest(j,:) - pop(j,:)) + c2*rand*(gbest - pop(j,:));
        %V(j,:) = GW*((fitness(j)/2000)^2+1)*rand*V(j,:) + c1*rand*(pbest(j,:) - pop(j,:)) + c2*rand*(gbest - pop(j,:));
        %V(j,:) = GW*V(j,:) + c1*rand*(pbest(j,:) - pop(j,:)) + c2*rand*(gbest - pop(j,:));
        %V(j,:) = 0.9*V(j   ,:) + c1*rand*(pbest(j,:) - pop(j,:)) + c2*rand*(gbest - pop(j,:));
        %V(j,:) = 0.9*1.0003^(-j)* V(j,:) + c1*rand*(pbest(j,:) - pop(j,:)) + c2*rand*(gbest - pop(j,:));
        %V(j,:) = (gbestfitness/(exp(-fitness(j))+1)+0.5)*rand*V(j,:) + c1*rand*(pbest(j,:) - pop(j,:)) + c2*rand*(gbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %粒子更新
        pop(j,:)=pop(j,:)+0.5*V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        %计算粒子适应度值
        Center=pop(j,1:3);
        SP=pop(j,4:6); 
        W=pop(j,7:9);
        Distance=dist(Center',SamIn);
        SPMat=repmat(SP',1,SamNum);%repmat具体作用
        UnitOut=radbas(Distance./SPMat);
        NetOut=W*UnitOut;%网络输出
        Error=SamOut-NetOut;%网络误差
        %SSE=sumsqr(Error);
        %fitness(j)=SSE;
        RMSE=(sumsqr(Error)/SamNum);
        fitness(j)=RMSE;
       % Center=pop(j,1:10);
       % SP=pop(j,11:20);
       % W=pop(j,21:30);
       % fitness(j)=fun(pop(j,:));
    end
    for j=1:popcount
        
        %个体最优更新
        if fitness(j) < pbestfitness(j)
            pbest(j,:) = pop(j,:);
            pbestfitness(j) = fitness(j); 
        end
        
        %群体最优更新
        if fitness(j) < gbestfitness
            gbest = pop(j,:);
            gbestfitness = fitness(j);
        end
    end
    gbesthistory=[gbesthistory,gbest];
    %mse(i)=gbestfitness;
    %将群体最优值赋给RBF参数
    Center=gbest(1,1:3);
    SP=gbest(1,4:6); 
    W=gbest(1,7:9);
    %Center=gbest(1,1:5);
    %SP=gbest(1,11:20); 
    % W=gbest(1,21:30);
     Distance=dist(Center',SamIn);
     SPMat=repmat(SP',1,SamNum);%repmat具体作用
     UnitOut=radbas(Distance./SPMat);
     NetOut=W*UnitOut;%网络输出
     Error=SamOut-NetOut;%网络误差
     %sse(i)=sumsqr(Error);
     mse(i)=(sumsqr(Error)/SamNum);
   % sse(i)=fun(gbest);
   %if sse(i)<E0,break,end 
end
toc;
% 测试 
Center=gbest(1,1:3);
SP=gbest(1,4:6); 
W=gbest(1,7:9);
TestDistance=dist(Center',TargetIn);
TesatSpreadsMat=repmat(SP',1,TargetSamNum);
TestHiddenUnitOut=radbas(TestDistance./TesatSpreadsMat);
TestNNOut=W*TestHiddenUnitOut;

%作图 分别在训练集和测试集上
subplot(1,2,1)
plot(1:length(NetOut),NetOut,'*',1:length(NetOut),SamOut,'o')
title('In Train data')
subplot(1,2,2)
plot(1:3,TestNNOut,'*',1:3,TargetOut,'o')
title('In Test data')
%求出误差 训练集和测试集
train_error=sum(abs(SamOut-NetOut))/length(SamOut);
test_error=sum(abs(TargetOut-TestNNOut))/length(TargetOut);

%  绘制学习误差曲线
figure
hold on
grid
%[xx,Num]=size(errhistory);
plot(mse,'k-');
legend('mse')