clc
clear all
close all
syms x

%get bounds of function and desirable precision
x_min=input("inter down bound:");
x_max=input("inter up bound:");
e=input("inter desirable precision:");

% define a g(x) function and calculating norm-inf of dg(x) in [x_min x_max] 
g =  (10*(x^4)*((exp(x)+exp(-1*x))/2));
dg = diff(g,x);
dg_func= matlabFunction(dg);
x_vals=linspace(x_min,x_max,1000) ;
dg_vals=abs(dg_func(x_vals));
norm_inf=max(dg_vals);

%plot of g(x) and gd(x)
ezplot(g,[x_min x_max]);
title('g(x)');
figure;
ezplot(dg,[x_min x_max]);
title('dg(x)');
figure;
%calculating h then number of fuzzy sets(n) with precision=e
h=e/norm_inf;
n=((x_max-x_min)/h)+1;


%make data between x_min and x_max with step h
xdata=linspace(x_min,x_max,n);
step=h;
x=x_min:step:x_max;
for i=1:n
 data(i)=(10*(xdata(i)^4)*((exp(xdata(i))+exp(-1*xdata(i)))/2));
end


% make membership function with h
a=x_min-h;
b=x_min;
c=x_min+h;
for i=1:n
 matrix(i,:)=trimf(x,[a b c]);
 plot(x,matrix(i,:))
 xlim([-0.01 0.01])
 hold on
 a=a+h;
 b=b+h;
 c=c+h;
end

%make f(x) function(fuzzy system)
num=0;
den=0;
for i=1:n
 num=num+data(i)*matrix(i,:);
 den=den+matrix(i,:);
end
fx=num./den;

% compare

g=(10.*(x.^4).*((exp(x)+exp(-1.*x))./2));
figure,plot(x,g,'b-',x,fx,'r.')
title('g(x) and f(x)')
legend('g(x)','f(x)')
figure,plot(x,fx,'g.')
title('f(x)')
figure,plot(x,abs(g-fx))
title('error')
