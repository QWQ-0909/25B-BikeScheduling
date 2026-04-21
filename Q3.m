% === Step 1: 读取数据 ===
data = readtable('最终归一化效率评分_压缩区间0.05到0.95.xlsx');
% === Step 2: 取出原始列（总满足率 和 缺口占比）===
R = data.总满足率; % 满足率 R_i
D = data.缺口占比; % 最大缺口占比 D_i
% === Step 3: 求最大最小值 ===
R_min = min(R);
R_max = max(R);
D_min = min(D);
D_max = max(D);
% === Step 4: 执行归一化 ===
R_norm = (R - R_min) / (R_max - R_min); % 归一化后的 R_i
D_norm = (D - D_min) / (D_max - D_min); % 归一化后的 D_i
% === Step 5: 最终效率评分 ===
E_score = R_norm - D_norm;
% === Step 6: 输出结果验证 ===
result = table(data.区域, R, D, R_norm, D_norm, E_score, ...
'VariableNames', {'区域', 'R原值', 'D原值', 'R归一化', 'D归一化', 'E归一化得分'});
disp(result)
% （可选）保存到 Excel 以对比验证
writetable(result, '验证_效率得分计算结果.xlsx');