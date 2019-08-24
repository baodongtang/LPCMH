clc;
clear;

load mirflickr5k;
fname = 'mirflickr5k';
name = strcat(fname,'.txt');
% make the training/test data zero-mean
I_te = bsxfun(@minus, I_te, mean(I_tr, 1));     %693*128
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));     %
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));     %693*10
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));


X_train = I_tr';
Y_train = T_tr';
Tr_label = L_tr;

X_test = I_te';
Y_test = T_te';
Te_label = L_te;

[d_I_tr,n_I_tr] = size(X_train);
[d_T_tr,n_T_tr] = size(Y_train);
[d_I_te,n_I_te] = size(X_test);
[d_T_te,n_T_te] = size(Y_test);
cc = unique(L_te);

S1 = constructW(X_train', struct('k', 5));
S2 = constructW(Y_train', struct('k', 5));
S1 = (S1+S1')/2;
S1 = max(S1,0);
S2 = (S2+S2')/2;
S2 = max(S2,0);
opt.S1 = S1;
opt.S2 = S2;

% opt.S1 = ones(n_I_tr);
% opt.S2 = ones(n_I_tr);

[ln,~] = size(L_tr);
R = zeros(ln);
for j = 1:ln
    a = L_tr(:,find(L_tr(j,:)));
    [~,m] = size(a);
    for i=1:m
        idx = find(a(:,i)==1);
        R(idx,j) = 1;
    end
end
opt.R = R;
 

c = length(unique(L_te));   %ÀàÊý
bits = [16,32,64,128];  %[16,32,64,128]
lambda1 = [1e-6,1e-5];
lambda2 = [1e-6,1e-5];
lambda3 = [1e-6,1e-5];



Tr_label = L_tr;
Te_label = L_te;


fin_result = cell(length(bits)*length(lambda1)*length(lambda2)*length(lambda3),1);
i_result = 1;

for bi = 1:length(bits)
    fid = fopen(name,'a+');
    fprintf(fid,'%5s\r\n',' ');
    fprintf(fid,'%5s\t','-------------------- bits =');
    fprintf(fid,'%8g\r\n',bits(bi));
    fclose(fid);
    for i = 1:length(lambda1)
        for j = 1:length(lambda2)
            for k = 1:length(lambda3)
                opt.lambda1 = lambda1(i);
                opt.lambda2 = lambda2(j);
                opt.lambda3 = lambda3(k);
                opt.bits = bits(bi);
                opt.maxItr = 1;
                [W, U, V, B, obj] = TrainMirfilckr(X_train, Y_train, Tr_label, opt);
                [result] = evaluation_mirflickr(U,V,X_test,Y_test,Tr_label,Te_label,B);
                fid = fopen(name,'a+');
                fprintf(fid,'%5s','lambda1 =');
                fprintf(fid,'%8.5g\t',lambda1(i));
                fprintf(fid,'%5s','lambda2 =');
                fprintf(fid,'%8.5g\t',lambda2(j));
                fprintf(fid,'%5s','lambda3 =');
                fprintf(fid,'%8.5g\t',lambda3(k));
%                 fprintf(fid,'%5s','I_Pre =');
%                 fprintf(fid,'%8.5g\t',result.I_Pre);
%                 fprintf(fid,'%5s','I_Rec =');
%                 fprintf(fid,'%8.5g\t',result.I_Rec);
                fprintf(fid,'%5s','I2T_MAP =');
                fprintf(fid,'%8.5g\t',result.I2T_MAP);
% %                 fprintf(fid,'%5s','T_Pre =');
%                 fprintf(fid,'%8.5g\t',result.T_Pre);
%                 fprintf(fid,'%5s','T_Rec =');
%                 fprintf(fid,'%8.5g\t',result.T_Rec);
                fprintf(fid,'%5s','T2I_MAP =');
                fprintf(fid,'%8.5g\t\n',result.T2I_MAP);
                fclose(fid);
                result.bits = bits(bi);
                result.lambda1 = lambda1(i);
                result.lambda2 = lambda2(j);
                result.lambda3 = lambda3(k);
                fin_result{i_result,1} = result;
                i_result = i_result + 1;
            end 
        end
    end
    name = strcat(fname,'.txt');
    fid = fopen(name,'a+');
    fprintf(fid,'%5s\t','-------------------- end time =');
    fprintf(fid,'%8s\r\n',datestr(now,31));
    fclose(fid);
end

save('fin_result','fin_result');
plot(1:length(obj),obj);