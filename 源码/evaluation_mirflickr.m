function result = evaluation_mirflickr(U,V,X_test,Y_test,Tr_label,Te_label,B)
% output I2T and T2I mAP

   hammRadius = 2;

   tUX = U'*X_test;
   tVY = V'*Y_test;

   tBX = compactbit(sign(tUX') >= 0);
   tBT = compactbit(sign(tVY') >= 0);
   B = compactbit(sign(B') >= 0);
   
   

   I_hammTrainTest = hammingDist(tBX, B)';

   % hamming ranking: MAP
   [~, I_HammingRank]=sort(I_hammTrainTest,1);
    I2T_MAP = cat_apcal(Tr_label,Te_label,I_HammingRank);
 %  I2T_MAP = perf_metric4Label( Tr_label,Te_label, I_hammTrainTest );

   
   
   T_hammTrainTest = hammingDist(tBT, B)';

%    [T_Pre, T_Rec] = evaluate_macro(I_cateTrainTest, T_Ret);  %相关的，查询的。
   % hamming ranking: MAP
   [~, T_HammingRank]=sort(T_hammTrainTest,1);    %对列排序
   T2I_MAP = cat_apcal(Tr_label,Te_label,T_HammingRank);
 % T2I_MAP = perf_metric4Label( Tr_label,Te_label, T_hammTrainTest );

   
   
%    result.I_Pre = I_Pre;
%    result.I_Rec = I_Rec;
   result.I2T_MAP = I2T_MAP;
%    result.T_Pre = T_Pre;
%    result.T_Rec = T_Rec;
   result.T2I_MAP = T2I_MAP;
end