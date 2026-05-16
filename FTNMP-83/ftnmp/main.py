import torch as tc
import graph_generate as gg
import numpy as np
import contract_exact as ce
import time
import EntropyBP as EBP
import EntropyFTNMP as EFT
import get_regions_new as grn
from tqdm import tqdm
import gc  # 用于内存回收

import EntropyFTNMP_old_1 as EFTO1
import EntropyFTNMP_old_2 as EFTO2

print("=" * 60)
print("        FTNBP 批量实验配置向导（仅需输入数字）")
print("Hello22366311")
print("=" * 60)

# ---------- 1. 实验次数 ----------
while True:
    try:
        NUM_EXPERIMENTS = int(input("\n请输入实验次数 (例如 50): "))
        if NUM_EXPERIMENTS > 0:
            break
        else:
            print("请输入大于0的整数。")
    except ValueError:
        print("输入无效，请输入一个整数。")

# ---------- 2. 节点数 ----------
while True:
    try:
        NUM_NODES = int(input("\n请输入图的节点数 (例如 200): "))
        if NUM_NODES > 0:
            break
        else:
            print("节点数必须大于0。")
    except ValueError:
        print("输入无效，请输入一个整数。")
print(f"选择的节点数 = {NUM_NODES}")

# ---------- 3. 密度 alpha ----------
alpha_presets = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\n请选择全局密度 alpha (标准值为 0.7)：")
print("0: 手动输入其他数值")
for i, val in enumerate(alpha_presets, 1):
    print(f"{i}: {val}")
while True:
    try:
        choice = int(input("请输入选项编号 (0-5): "))
        if choice == 0:
            alpha = float(input("请输入自定义 alpha 值 (0~1之间): "))
            if 0 < alpha <= 1:
                break
            else:
                print("alpha 应在 (0, 1] 范围内。")
        elif 1 <= choice <= len(alpha_presets):
            alpha = alpha_presets[choice - 1]
            break
        else:
            print(f"输入无效，请输入 0-{len(alpha_presets)} 之间的数字。")
    except ValueError:
        print("输入无效，请输入数字。")
print(f"选择的 alpha = {alpha}")

# ---------- 4. 聚类数 n_cl ----------
ncl_options = list(range(21))  # 0 到 20
print("\n请选择聚类数 n_cl (0-20)：")
for i, val in enumerate(ncl_options, 1):
    print(f"{i}: {val}")
while True:
    try:
        choice = int(input("请输入选项编号 (1-21): "))
        if 1 <= choice <= len(ncl_options):
            n_cl = ncl_options[choice - 1]
            break
        else:
            print(f"输入无效，请输入 1-{len(ncl_options)} 之间的数字。")
    except ValueError:
        print("输入无效，请输入一个整数。")
print(f"选择的 n_cl = {n_cl}")

# ---------- 5. 局部密集选项 ----------
print("\n是否启用局部密集子图？")
print("1: 是（使用固定参数 alpha_local=[2.5,3.2], min_cl=8, max_cl=12）")
print("0: 否")
while True:
    try:
        local_dense_choice = int(input("请输入选项编号 (0 或 1): "))
        if local_dense_choice in (0, 1):
            use_local_dense = bool(local_dense_choice)
            break
        else:
            print("输入无效，请输入 0 或 1。")
    except ValueError:
        print("输入无效，请输入一个整数。")

if use_local_dense:
    alpha_local = [2.5, 3.2]
    min_cl, max_cl = 8, 12
    print("已启用局部密集子图。")
else:
    alpha_local = None
    min_cl = max_cl = None
    print("未启用局部密集子图。")

# ---------- 6. R 值组合 ----------
r_options = {
    1: [4],
    2: [4, 6],
    3: [4, 6, 8],
    4: [4, 6, 8, 10],
    5: [4, 6, 8, 10, 12]
}
print("\n请选择要计算的 R 值组合：")
for key, val in r_options.items():
    print(f"{key}: {val}")
while True:
    try:
        choice = int(input("请输入选项编号 (1-5): "))
        if choice in r_options:
            RRR = r_options[choice]
            break
        else:
            print("输入无效，请输入 1-5 之间的数字。")
    except ValueError:
        print("输入无效，请输入一个整数。")
print(f"将计算 R 值: {RRR}")

# ---------- 7. cavity2 初始 R 值选择 ----------
print("\n请选择用于生成初始 cavity2 的基础 FTNBP 的 R 值：")
print("1: R=2")
print("2: R=4")
while True:
    try:
        r_init_choice = int(input("请输入选项编号 (1 或 2): "))
        if r_init_choice == 1:
            R_INIT = 2
            break
        elif r_init_choice == 2:
            R_INIT = 4
            break
        else:
            print("输入无效，请输入 1 或 2。")
    except ValueError:
        print("输入无效，请输入一个整数。")
print(f"初始 cavity2 将由 FTNBP (R={R_INIT}) 计算得到。")

# ---------- 8. 实验版本组合 ----------
print("\n请选择要运行的算法版本组合：")
print("1: 仅对照组（原版 FTNBP）")
print("2: 对照组 + cavity优化")
print("3: 对照组 + cavity优化+skip")
print("4: 全部三个版本（原版 + cavity优化 + cavity优化+skip）")
print("5: 仅 cavity优化+skip（只运行 skip 版本）")   # 新增选项
while True:
    try:
        run_mode_choice = int(input("请输入选项编号 (1-5): "))
        if 1 <= run_mode_choice <= 5:
            if run_mode_choice == 1:
                RUN_MODE = 'base'
            elif run_mode_choice == 2:
                RUN_MODE = 'cavity'
            elif run_mode_choice == 3:
                RUN_MODE = 'skip'
            elif run_mode_choice == 4:
                RUN_MODE = 'all'
            elif run_mode_choice == 5:
                RUN_MODE = 'skip_only'
            break
        else:
            print("输入无效，请输入 1-5 之间的数字。")
    except ValueError:
        print("输入无效，请输入一个整数。")

mode_desc = {
    'base': '仅原版',
    'cavity': '原版 + cavity优化',
    'skip': '原版 + cavity优化+skip',
    'all': '全部三个版本',
    'skip_only': '仅 cavity优化+skip'          # 新增描述
}
print(f"已选择实验模式: {mode_desc[RUN_MODE]}")

# ---------- 9. 确认开始 ----------
print("\n" + "=" * 60)
print("所有参数已设置完毕。")
print(f"实验次数: {NUM_EXPERIMENTS}")
print(f"节点数: {NUM_NODES}")
print(f"密度 alpha: {alpha}")
print(f"聚类数 n_cl: {n_cl}")
print(f"局部密集: {'是' if use_local_dense else '否'}")
print(f"R 值列表: {RRR}")
print(f"初始 cavity2 来源: R={R_INIT}")
print(f"运行模式: {mode_desc[RUN_MODE]}")
print("=" * 60)

while True:
    confirm = input("是否开始运行实验？(输入 1 开始，输入 0 退出): ")
    if confirm == '1':
        print("\n实验开始...\n")
        break
    elif confirm == '0':
        print("程序已退出。")
        exit()
    else:
        print("请输入 1 或 0。")

# 固定参数
device = 'cuda' if tc.cuda.is_available() else 'cpu'
R_subregion = 100

# 收集结果
all_times = []
all_steps = []
all_entropies = []
all_errors = []

seed = 0
oom_count = 0

for exp_idx in tqdm(range(NUM_EXPERIMENTS), desc="实验进度"):
    try:
        # 生成随机图
        clauses, tendencies = gg.double_random_generate(
            NUM_NODES, alpha, n_cl, alpha_local, min_cl, max_cl, seed
        )
        G_fac, max_item = gg.G_generator_fac(clauses)

        # 精确求解
        configrite_number_qu, sgn, T_exact = ce.local_contraction_ds(
            G_fac, clauses, tendencies, max_item, 25, device
        )
        entropy_exact_qu = tc.log(configrite_number_qu) + sgn * tc.log(tc.tensor(2))
        entropy_exact_qu = tc.where(
            entropy_exact_qu.isnan() | entropy_exact_qu.isneginf(),
            tc.tensor(0.0).to(device), entropy_exact_qu
        )

        if entropy_exact_qu == 0:
            seed += 1
            continue

        # 初始化结果列表
        times = [T_exact]
        steps = [0]
        entropies = [entropy_exact_qu.item()]
        errors = [0.0]

        print(f"\n{'='*60}")
        print(f"Experiment {exp_idx+1}/{NUM_EXPERIMENTS} (seed={seed})")
        print(f"Exact entropy: {entropy_exact_qu.item():.10f}, time: {T_exact:.4f}s")
        print()

        # BP
        t0 = time.time()
        entropy_BP, step_BP = EBP.EntropyBP(clauses, tendencies)
        entropy_BP = tc.nan_to_num(entropy_BP, nan=0.0)
        t_bp = time.time() - t0
        times.append(t_bp)
        steps.append(step_BP)
        entropies.append(entropy_BP.item())
        error_bp = (abs(entropy_BP - entropy_exact_qu) / entropy_exact_qu).item()
        errors.append(error_bp)
        print(f"BP            : entropy={entropy_BP.item():.10f}, error={error_bp:.6e}, steps={step_BP}, time={t_bp:.4f}s")
        print()

        # 清理缓存，消除顺序影响
        if device == 'cuda':
            tc.cuda.synchronize()
            tc.cuda.empty_cache()
        gc.collect()

        # 计算初始 cavity2（基于 R_INIT）
        devides_init, _, region_info_init, single_list_init = grn.get_regions_z(
            G_fac, clauses, max_item, R_INIT, R_subregion
        )
        Nv_init, boundaries_init = grn.devides_to_Nv(G_fac, devides_init, region_info_init)
        devide_bound_init, degree_node_init = grn.devides_others(G_fac, devides_init)

        t_init = time.time()
        entropy_init, step_init, cavity2 = EFT.EntropyZ_new(
            G_fac, clauses, tendencies, devides_init, devide_bound_init, region_info_init,
            single_list_init, max_item, Nv_init, boundaries_init, degree_node_init, device, None, None
        )
        entropy_init = tc.nan_to_num(entropy_init, nan=0.0)
        t_init_elapsed = time.time() - t_init
        times.append(t_init_elapsed)
        steps.append(step_init)
        entropies.append(entropy_init.item())
        error_init = (abs(entropy_init - entropy_exact_qu) / entropy_exact_qu).item()
        errors.append(error_init)
        print(f"FTNBP (R={R_INIT}) init: entropy={entropy_init.item():.10f}, error={error_init:.6e}, steps={step_init}, time={t_init_elapsed:.4f}s")
        print()

        # 根据 RUN_MODE 确定要运行的版本
        version_configs = {
            'base': ('原版', EFTO1),
            'cavity': ('cavity优化', EFTO2),
            'skip': ('cavity优化+skip', EFT)
        }
        if RUN_MODE == 'base':
            versions_to_run = [version_configs['base']]
        elif RUN_MODE == 'cavity':
            versions_to_run = [version_configs['base'], version_configs['cavity']]
        elif RUN_MODE == 'skip':
            versions_to_run = [version_configs['base'], version_configs['skip']]
        elif RUN_MODE == 'skip_only':    # 新增
            versions_to_run = [version_configs['skip']]
        else:  # 'all'
            versions_to_run = [version_configs['base'], version_configs['cavity'], version_configs['skip']]

        # 运行各版本
        for ver_name, module in versions_to_run:
            for R_region in RRR:
                cavity2_copy = cavity2.clone()
                devides, _, region_info, single_list = grn.get_regions_z(
                    G_fac, clauses, max_item, R_region, R_subregion
                )
                Nv, boundaries = grn.devides_to_Nv(G_fac, devides, region_info)
                devide_bound, degree_node = grn.devides_others(G_fac, devides)
                t_start = time.time()
                if module == EFTO1:
                    entropy_val, step_val, _ = module.EntropyZ_new(
                        G_fac, clauses, tendencies, devides, devide_bound, region_info,
                        single_list, max_item, Nv, boundaries, degree_node, device, None, None
                    )
                else:
                    entropy_val, step_val, _ = module.EntropyZ_new(
                        G_fac, clauses, tendencies, devides, devide_bound, region_info,
                        single_list, max_item, Nv, boundaries, degree_node, device, None, cavity2_copy
                    )
                entropy_val = tc.nan_to_num(entropy_val, nan=0.0)
                elapsed = time.time() - t_start

                times.append(elapsed)
                steps.append(step_val)
                entropies.append(entropy_val.item())
                error_val = (abs(entropy_val - entropy_exact_qu) / entropy_exact_qu).item()
                errors.append(error_val)
                print(f"FTNBP_{R_region}_{ver_name}: entropy={entropy_val.item():.10f}, error={error_val:.6e}, steps={step_val}, time={elapsed:.4f}s")
                print()

        all_times.append(times)
        all_steps.append(steps)
        all_entropies.append(entropies)
        all_errors.append(errors)
        print(f"Experiment {exp_idx+1} completed.\n{'='*60}")

    except RuntimeError as e:
        if 'out of memory' in str(e):
            oom_count += 1
            print(f"实验 {exp_idx} 因显存不足跳过。")
            continue
        else:
            raise e
    seed += 1

# ---------- 汇总与输出 ----------
if len(all_times) == 0:
    print("没有成功完成的实验，无法输出结果。")
    exit()

all_times = np.array(all_times)
all_steps = np.array(all_steps)
all_entropies = np.array(all_entropies)
all_errors = np.array(all_errors)

avg_times = np.mean(all_times, axis=0)
avg_steps = np.mean(all_steps, axis=0)
avg_entropies = np.mean(all_entropies, axis=0)
avg_errors = np.mean(all_errors, axis=0)

# 构建方法名称列表
fixed_methods = ['exact', 'BP', f'FTNBP_{R_INIT}_init']
if RUN_MODE == 'base':
    versions = ['原版']
elif RUN_MODE == 'cavity':
    versions = ['原版', 'cavity优化']
elif RUN_MODE == 'skip':
    versions = ['原版', 'cavity优化+skip']
elif RUN_MODE == 'skip_only':
    versions = ['cavity优化+skip']          # 新增
else:
    versions = ['原版', 'cavity优化', 'cavity优化+skip']

method_names = fixed_methods[:]
for ver in versions:
    for r in RRR:
        method_names.append(f'FTNBP_{r}_{ver}')

if len(method_names) > len(avg_times):
    method_names = method_names[:len(avg_times)]

print("\n" + "=" * 100)
print(f"实验完成统计：成功 {len(all_times)} 次，显存不足跳过 {oom_count} 次")
print("=" * 100)

print("\n【实验配置】")
print(f"  节点数 (NUM_NODES)           = {NUM_NODES}")
print(f"  全局密度 (alpha)             = {alpha}")
print(f"  聚类数 (n_cl)                = {n_cl}")
print(f"  局部密集子图 (use_local_dense) = {use_local_dense}")
print(f"  计算的 R 值列表 (RRR)        = {RRR}")
print(f"  初始 cavity2 的 R (R_INIT)   = {R_INIT}")
print(f"  运行模式 (RUN_MODE)          = {RUN_MODE} ({mode_desc[RUN_MODE]})")
print("=" * 100)

print("平均结果汇总：")
print("Method                      Time (s)           Steps      Entropy          Rel. Error")
print("-" * 100)

print(f"{method_names[0]:<25} {avg_times[0]:<20.10f} {'-':<12} {avg_entropies[0]:<18.10f} {avg_errors[0]:<15.10f}")

for i in range(1, len(method_names)):
    step_str = f"{avg_steps[i]:.4f}" if i < len(avg_steps) else '-'
    print(f"{method_names[i]:<25} {avg_times[i]:<20.10f} {step_str:<12} {avg_entropies[i]:<18.10f} {avg_errors[i]:<15.10f}")

with open("FTNMP_results.txt", "w") as f:
    f.write(f"参数：节点数={NUM_NODES}, alpha={alpha}, n_cl={n_cl}, 局部密集={use_local_dense}, R={RRR}, 初始R={R_INIT}, 模式={RUN_MODE}\n")
    f.write("Method                      Time (s)           Steps      Entropy          Rel. Error\n")
    f.write("-" * 100 + "\n")
    f.write(f"{method_names[0]:<25} {avg_times[0]:<20.10f} {'-':<12} {avg_entropies[0]:<18.10f} {avg_errors[0]:<15.10f}\n")
    for i in range(1, len(method_names)):
        step_str = f"{avg_steps[i]:.4f}" if i < len(avg_steps) else '-'
        f.write(f"{method_names[i]:<25} {avg_times[i]:<20.10f} {step_str:<12} {avg_entropies[i]:<18.10f} {avg_errors[i]:<15.10f}\n")

print("\n结果已保存至 FTNMP_results.txt")