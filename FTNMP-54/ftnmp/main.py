import torch as tc
import graph_generate as gg
import numpy as np
import contract_exact as ce
import time
import EntropyBP as EBP
import EntropyFTNMP as EFT
import get_regions_new as grn
from tqdm import tqdm


# ===== 方法1：整体差异（L1）=====
def cavity_diff(cav1, cav2):
    return tc.mean(tc.abs(cav1 - cav2)).item()

# ===== 方法2：边级差异=====
def cavity_edge_diff(cav1, cav2, threshold=1e-2):
    n = cav1.shape[0]
    big_diff = []
    for i in range(n):
        for j in range(n):
            d = tc.sum(tc.abs(cav1[i][j] - cav2[i][j])).item()
            if d > threshold:
                big_diff.append((i, j, d))
    return big_diff

# ===== 方法3：熵=====
def cavity_entropy(cav):
    eps = 1e-12
    cav = cav + eps
    return -tc.sum(cav * tc.log(cav), dim=2).mean().item()


print("Hello 54511")

device = 'cuda'
R_subregion = 100
error_FTNBP_6=[]
error_BP = []
file = open('FTNMP.txt','w')
#times = []
Error =[]
Entropy =[]
times =[]
steps =[]
seed =0
oom = 0
for i in tqdm(range(0,4)):
    t = []
    Ey = []
    Er = []
    step=[]
    nodes = 100
    alpha =0.7
    n_cl = 0
    alpha_local=[2.5,3.2]
    max_cl = 12
    min_cl = 8
    clauses,tendencies = gg.double_random_generate(nodes,alpha,n_cl,alpha_local,min_cl,max_cl,seed)
    print('nodes:',nodes,'alpha_grobal:',alpha,'number of cluster:',n_cl,'alpha_local:',alpha_local,'Maximum points:',max_cl,'Minimum points:',min_cl,file = file)
    print('clauses=',clauses,file =file)
    print('tendencies=',tendencies,file = file)
    print('alpha:',len(clauses)/nodes,file = file)
    G_fac,max_item =gg.G_generator_fac(clauses)
    nodes = list(range(max_item+1))
    print(f"Step {i}  begin:")
    try:
       
        configrite_number_qu,sgn,T =ce.local_contraction_ds(G_fac,clauses,tendencies,max_item,25,device)
        entropy_exact_qu = tc.log(configrite_number_qu)+sgn*tc.log(tc.tensor(2))
        entropy_exact_qu = tc.where(entropy_exact_qu.isnan() | entropy_exact_qu.isneginf(), tc.tensor(0.0).to(device), entropy_exact_qu)
        #print( 'time_opt:',T,entropy_exact_qu)
        t.append(T)
        if entropy_exact_qu !=0:
            Ey.append(entropy_exact_qu.item())
            Er.append(0)
        else:
            continue
        ########################################################################################################################
        #print('\n','BP:')
        t5 = time.time()
        entropy_BP,step_BP = EBP.EntropyBP(clauses,tendencies)
        entropy_BP = tc.nan_to_num(entropy_BP, nan=0.0)
        t6 = time.time()
        #print('BP   :',entropy_BP)
        t.append(t6-t5)
        step.append(step_BP)
        if entropy_exact_qu !=0:
            Ey.append(entropy_BP.item())
            Er.append((abs(entropy_BP-entropy_exact_qu)/entropy_exact_qu).item())
        ########################################################################################################################

        # 第一阶段单次 FTNBP
        cavity_dict = {}
        for R in [2, 4, 6, 8, 10]:
            R_region = R
            devides, subdevides, region_info, single_list = grn.get_regions_z(
                G_fac, clauses, max_item, R_region, R_subregion
            )
            Nv, boundaries = grn.devides_to_Nv(G_fac, devides, region_info)
            devide_bound, degree_node = grn.devides_others(G_fac, devides)

            t_start = time.time()
            entropy, step_val, cavity = EFT.EntropyZ_new(
                G_fac, clauses, tendencies,
                devides, devide_bound, region_info,
                single_list, max_item, Nv, boundaries,
                degree_node, device, file, None
            )
            t_end = time.time()

            locals()[f'entropy_FTNBP_{R}_0'] = entropy
            locals()[f'step_{R}_0'] = step_val

            t.append(t_end - t_start)
            step.append(step_val)

            if entropy_exact_qu != 0:
                Ey.append(entropy.item())
                Er.append((abs(entropy - entropy_exact_qu) / entropy_exact_qu).item())

            # 存 cavity
            cavity_dict[(R, 0)] = cavity

        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        '''
        # 比较不同 R 的单次 cavity 结果（以 R=2 为基准）
        print("\n" + "=" * 60)
        print("Comparing cavities from single FTNBP: R=2 vs others")
        print("=" * 60)

        base_R = 2
        base_cavity = cavity_dict[(base_R, 0)]

        for R in [4, 6, 8, 10]:
            cav = cavity_dict[(R, 0)]
            diff_abs = tc.abs(cav - base_cavity)

            # 整体 L1 差异
            diff_total = diff_abs.sum().item()
            print(f"[R={R} vs R={base_R}] 整体 L1 差异: {diff_total:.6e}")

            # 超阈值元素数量
            thresholds = [0.03, 0.05, 0.1, 0.15, 0.2]
            for th in thresholds:
                count = (diff_abs > th).sum().item()
                print(f"    超过阈值 {th} 的元素数量: {count}")

            # 差值统计 + 直方图
            bins = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 1.0]
            print(f"    差值最大值: {diff_abs.max().item():.6e}, 平均值: {diff_abs.mean().item():.6e}")
            hist = tc.histc(diff_abs.flatten(), bins=len(bins) - 1, min=bins[0], max=bins[-1])
            print(f"    差值分布直方图: {hist.tolist()}")
            print()

        '''
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        for R in [2, 4, 6, 8, 10]:
            R_region = R
            devides, subdevides, region_info, single_list = grn.get_regions_z(
                G_fac, clauses, max_item, R_region, R_subregion
            )
            Nv, boundaries = grn.devides_to_Nv(G_fac, devides, region_info)
            devide_bound, degree_node = grn.devides_others(G_fac, devides)

            for it in [2, 4, 6, 8, 10]:
                cavity_input = cavity_dict[(it, 0)].clone().detach()
                cavity_input2 = cavity_dict[(it, 0)].clone().detach()

                t_start = time.time()
                entropy, step_val, cavity = EFT.EntropyZ_new(
                    G_fac, clauses, tendencies,
                    devides, devide_bound, region_info,
                    single_list, max_item, Nv, boundaries,
                    degree_node, device, file, cavity_input2
                )
                t_end = time.time()


                locals()[f'entropy_FTNBP_{R}_{it}'] = entropy
                locals()[f'step_{R}_{it}'] = step_val

                t.append(t_end - t_start)
                step.append(step_val)

                Ey.append(entropy.item())
                Er.append((abs(entropy - entropy_exact_qu) / entropy_exact_qu).item())

                # 存 cavity
                #cavity_dict[(R, it)] = cavity

                ########################################################################################################################
                ########################################################################################################################
                ########################################################################################################################


                # ===== 方法1：整体L1差异 =====
                diff = tc.abs(cavity - cavity_input).sum().item()
                print(f"[R={R}, iter={it}] 整体L1差异: {diff:.6e}")

                # ===== 方法2：超阈值边差异 =====

                thresholds = [0.03,0.05, 0.1, 0.15, 0.2]  # 阈值列表
                diff_abs = tc.abs(cavity - cavity_input)
                #for th in thresholds:
                #    count = (diff_abs > th).sum().item()
                #    print(f"[R={R}, iter={it}] 超过阈值 {th} 的元素数量: {count}")


                # ===== 方法3：差值统计 + 直方图 =====
                diff_np = diff_abs.cpu().numpy().flatten()
                bins = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5,0.6, 1.0]
                hist, bin_edges = np.histogram(diff_np, bins=bins)

                print(f"[R={R}, iter={it}] 差值分布直方图:")
                for i in range(len(hist)):
                    print(f"  {bin_edges[i]:.1e} ~ {bin_edges[i + 1]:.1e}: {hist[i]}")

                # ===== 方法4：变化节点数量统计分布 =====
                threshold_node = 1e-2  # 节点变化判定阈值
                edge_changed = (diff_abs > threshold_node).any(dim=2)  # shape (n,n)
                node_impact = edge_changed.sum(dim=0)  # 每个中心节点受影响的边数
                unique_vals, counts = tc.unique(node_impact, return_counts=True)
                print(f"[R={R}, iter={it}] 节点变化次数分布 (变化次数:节点个数):")
                for val, cnt in zip(unique_vals.tolist(), counts.tolist()):
                    print(f"  {val}: {cnt}")

                # ===== 方法5：boundary vs inner =====
                boundary_edges = 0
                inner_edges = 0
                for i in range(edge_changed.shape[0]):
                    for j in range(edge_changed.shape[1]):
                        if edge_changed[i, j]:
                            if j in boundaries[i]:
                                boundary_edges += 1
                            else:
                                inner_edges += 1
                print(f"[R={R}, iter={it}] boundary变化: {boundary_edges}, inner变化: {inner_edges}")

        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################


        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################

        # ==================== 结果汇总打印（包含单次+递进共30个FTNBP） ====================
        print('\n' + '=' * 80)
        print("Step {} final results:".format(i))
        print('=' * 80)

        # 精确熵
        print(f"{'Exact':<20} entropy: {entropy_exact_qu.item():.10f}")
        print(f"{'Exact':<20} error  : 0.0")
        print(f"{'Exact':<20} time   : {t[0]:.6f} s")
        print('-' * 80)

        # BP
        print(f"{'BP':<20} entropy: {entropy_BP.item():.10f}")
        print(f"{'BP':<20} error  : {abs(entropy_exact_qu.item() - entropy_BP.item()):.10f}")
        print(f"{'BP':<20} time   : {t[1]:.6f} s")
        print(f"{'BP':<20} steps  : {step[0]}")
        print('-' * 80)

        # 单次 FTNBP (R=2,4,6,8,10, iter=0)
        single_R = [2, 4, 6, 8, 10]
        time_idx = 2
        step_idx = 1
        for R in single_R:
            entropy_var = f'entropy_FTNBP_{R}_0'
            step_var = f'step_{R}_0'
            method_name = f'FTNBP_{R}_0'
            if entropy_var in locals() and step_var in locals():
                entropy_val = locals()[entropy_var].item()
                error_val = abs(entropy_val - entropy_exact_qu.item()) / entropy_exact_qu.item()
                time_val = t[time_idx] if time_idx < len(t) else float('nan')
                step_val = locals()[step_var] if step_idx < len(step) else 'N/A'
                print(f"{method_name:<20} entropy: {entropy_val:.10f}")
                print(f"{method_name:<20} error  : {error_val:.10f}")
                print(f"{method_name:<20} time   : {time_val:.6f} s")
                print(f"{method_name:<20} steps  : {step_val}")
            else:
                print(f"{method_name:<20} 数据缺失（变量未定义）")
            time_idx += 1
            step_idx += 1
            print('-' * 80)

        # 递进 FTNBP (R=2,4,6,8,10, iter=2,4,6,8,10)
        R_list = [2, 4, 6, 8, 10]
        iter_list = [2, 4, 6, 8, 10]
        for R in R_list:
            for it in iter_list:
                entropy_var = f'entropy_FTNBP_{R}_{it}'
                step_var = f'step_{R}_{it}'
                method_name = f'FTNBP_{R}_{it}'
                if entropy_var in locals() and step_var in locals():
                    entropy_val = locals()[entropy_var].item()
                    error_val = abs(entropy_val - entropy_exact_qu.item()) / entropy_exact_qu.item()
                    time_val = t[time_idx] if time_idx < len(t) else float('nan')
                    step_val = locals()[step_var] if step_idx < len(step) else 'N/A'
                    print(f"{method_name:<20} entropy: {entropy_val:.10f}")
                    print(f"{method_name:<20} error  : {error_val:.10f}")
                    print(f"{method_name:<20} time   : {time_val:.6f} s")
                    print(f"{method_name:<20} steps  : {step_val}")
                else:
                    print(f"{method_name:<20} 数据缺失（变量未定义）")
                time_idx += 1
                step_idx += 1
                print('-' * 80)

        # 数据积累部分保持不变
        times.append(t)
        steps.append(step)
        if len(Ey) != 0:
            Entropy.append(Ey)
            Error.append(Er)

        Err = np.array(Error)
        column_means_Er = np.mean(Err, axis=0)

        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        ########################################################################################################################
        # ==================== 文件写入：详细结果 ====================
        # 写入当前实例的精确熵和BP
        print('exact  :', entropy_exact_qu.item(), file=file)
        print('BP     :', entropy_BP.item(), file=file)

        # 写入单次 FTNBP 的熵值 (R=2,4,6,8,10, iter=0)
        for R in [2, 4, 6, 8, 10]:
            entropy_var = f'entropy_FTNBP_{R}_0'
            if entropy_var in locals():
                print(f'FTNBP_{R}_0:', locals()[entropy_var].item(), file=file)
            else:
                print(f'FTNBP_{R}_0:', 'NaN', file=file)

        # 写入递进 FTNBP 的熵值 (R=2,4,6,8,10, iter=2,4,6,8,10)
        for R in [2, 4, 6, 8, 10]:
            for it in [2, 4, 6, 8, 10]:
                entropy_var = f'entropy_FTNBP_{R}_{it}'
                if entropy_var in locals():
                    print(f'FTNBP_{R}_{it}:', locals()[entropy_var].item(), file=file)
                else:
                    print(f'FTNBP_{R}_{it}:', 'NaN', file=file)

        # 写入BP的相对误差
        print('error_BP   :', (abs(entropy_exact_qu - entropy_BP) / entropy_exact_qu).item(), file=file)

        # 写入单次 FTNBP 的相对误差
        for R in [2, 4, 6, 8, 10]:
            entropy_var = f'entropy_FTNBP_{R}_0'
            if entropy_var in locals():
                entropy_val = locals()[entropy_var]
                error_val = (abs(entropy_exact_qu - entropy_val) / entropy_exact_qu).item()
                print(f'error_FTNBP_{R}_0:', error_val, file=file)
            else:
                print(f'error_FTNBP_{R}_0:', 'NaN', file=file)

        # 写入递进 FTNBP 的相对误差
        for R in [2, 4, 6, 8, 10]:
            for it in [2, 4, 6, 8, 10]:
                entropy_var = f'entropy_FTNBP_{R}_{it}'
                if entropy_var in locals():
                    entropy_val = locals()[entropy_var]
                    error_val = (abs(entropy_exact_qu - entropy_val) / entropy_exact_qu).item()
                    print(f'error_FTNBP_{R}_{it}:', error_val, file=file)
                else:
                    print(f'error_FTNBP_{R}_{it}:', 'NaN', file=file)

        # ==================== 累计统计 ====================
        print('_____________________________________ERROR_________________________________________', file=file)
        Err = np.array(Error)  # 形状 (num_instances, 32)
        print(Error, file=file)
        column_means_Er = np.mean(Err, axis=0)
        # 生成列标题（32列：EXACT, BP, 5个单次, 25个递进）
        single_methods = [f'FTNBP_{R}_0' for R in [2, 4, 6, 8, 10]]
        iter_methods = [f'FTNBP_{R}_{it}' for R in [2, 4, 6, 8, 10] for it in [2, 4, 6, 8, 10]]
        methods = ['EXACT', 'BP'] + single_methods + iter_methods
        header_Er = '---'.join(methods)
        print(f'{header_Er}----Errorbar:', '\n', column_means_Er, file=file)
        max_valuesr = np.max(Err, axis=0)
        min_valuesr = np.min(Err, axis=0)
        resultr = list(zip(min_valuesr, max_valuesr))
        print(f'{header_Er}----Error(min,max):', '\n', resultr, file=file)

        print('_________________________________instance entropy____________________________________', file=file)
        print(Entropy, file=file)
        E = np.array(Entropy)  # 形状 (num_instances, 32)
        column_means_E = np.mean(E, axis=0)
        print(f'{header_Er}----Sbar:', '\n', column_means_E, file=file)
        max_values = np.max(E, axis=0)
        min_values = np.min(E, axis=0)
        result = list(zip(min_values, max_values))
        print(f'{header_Er}----S(min,max):', '\n', result, file=file)

        print('_________________________________average time_______________________________________', file=file)
        print(times, file=file)
        T = np.array(times)  # 形状 (num_instances, 32)
        column_means_T = np.mean(T, axis=0)
        print(f'{header_Er}----Tbar:', '\n', column_means_T, file=file)
        max_valuest = np.max(T, axis=0)
        min_valuest = np.min(T, axis=0)
        resultt = list(zip(min_valuest, max_valuest))
        print(f'{header_Er}----T(min,max):', '\n', resultt, file=file)

        print('_______________________________________steps_______________________________________', file=file)
        print(steps, file=file)
        STEP = np.array(steps)  # 形状 (num_instances, 31)  无exact列
        # steps的列标题：BP, 5个单次, 25个递进
        methods_steps = ['BP'] + single_methods + iter_methods
        header_steps = '---'.join(methods_steps)
        column_means_steps = np.mean(STEP, axis=0)
        print(f'{header_steps}----Stepsbar:', '\n', column_means_steps, file=file)
        max_valuesp = np.max(STEP, axis=0)
        min_valuesp = np.min(STEP, axis=0)
        resultp = list(zip(min_valuesp, max_valuesp))
        print(f'{header_steps}----Steps(min,max):', '\n', resultp, file=file)

        print('\n\n', file=file)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            oom += 1
            print(f"Step {i} skipped due to OOM error.")
        # 若希望跳过继续运行，可将 raise 改为 continue 并手动增加 seed
        raise






    ########################################################################################################################
    ########################################################################################################################
    seed = seed + 1
print('oom:',oom,file = file)




print('\n' + '=' * 100)
print("Summary Table (32 x 4) -- AVERAGE")
print('=' * 100)

methods = (
    ['Exact', 'BP'] +
    [f'FTNBP_{R}_0' for R in [2,4,6,8,10]] +
    [f'FTNBP_{R}_{it}' for R in [2,4,6,8,10] for it in [2,4,6,8,10]]
)

# ===== 用累计数据 =====
E = np.array(Entropy)   # (N,32)
Err = np.array(Error)   # (N,32)
T = np.array(times)     # (N,32)
STEP = np.array(steps)  # (N,31)

# ===== 求平均 =====
E_mean = np.mean(E, axis=0)
Err_mean = np.mean(Err, axis=0)
T_mean = np.mean(T, axis=0)
STEP_mean = np.mean(STEP, axis=0)

# ===== 打印 =====
print(f"{'Method':<15} {'Entropy':<15} {'Error':<15} {'Time(s)':<12} {'Steps':<10}")
print('-'*100)

for i, method in enumerate(methods):

    entropy_val = E_mean[i]
    error_val = Err_mean[i]
    time_val = T_mean[i]

    if method == 'Exact':
        step_val = 'N/A'
    else:
        step_val = STEP_mean[i-1]  # 注意偏移

    print(f"{method:<15} {entropy_val:<15.10f} {error_val:<15.10f} {time_val:<12.6f} {step_val}")