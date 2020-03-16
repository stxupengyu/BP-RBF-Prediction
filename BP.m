clear
clc
%创建训练样本输入集
load data;
%归一化
data1=data';
data=mapminmax(data1,0,1);
data=data';
num=2;%对应四个特征
%建立训练集测试集
x_train=[data(1:48,1).';data(1:48,2).';data(1:48,3).';data(1:48,4).'];
x_test=[data(49:51,1).';data(49:51,2).';data(49:51,3).';data(49:51,4).'];
y_train=[data(2:49,num).'];
y_test=[data(50:52,num).'];
%创建BP神经网络
%创建网络
net=newff(minmax(x_train),[4,1],{'tansig','purelin'},'trainlm');%隐层神经元个数，输出层神经元个数
%设置训练次数
net.trainParam.epochs = 100;
%设置收敛误差
net.trainParam.goal=0.001;
%训练网络
[net,tr]=train(net,x_train,y_train);
%在训练集和测试集上的表现
y_train_predict=sim(net,x_train);
y_test_predict=sim(net,x_test);
%作图 分别在训练集和测试集上
subplot(1,2,1)
plot(1:length(y_train_predict),y_train_predict,'*',1:length(y_train_predict),y_train,'o')
title('In Train data')
subplot(1,2,2)
plot(1:3,y_test_predict,'*',1:3,y_test,'o')
title('In Test data')
%求出误差 训练集和测试集
train_error=sum(abs(y_train_predict- y_train))/length(y_train);
test_error=sum(abs(y_test_predict- y_test))/length(y_test);





