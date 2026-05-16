import torch as tc
import networkx as nx
import copy
import opt_einsum as oe
import time
import numpy as np


def G_generator_fac(clauses):
    G = nx.Graph()
    if len(clauses) == 0:
        max_item = 0
    else:
        max_item = max(max(row) for row in clauses if len(row) != 0)
        for clause_id in range(len(clauses)):
            clause = clauses[clause_id]
            if len(clause) != 0:
                qubit2 = max_item + clause_id + 1
                for qubit1_id in range(len(clause)):
                    qubit1 = clause[qubit1_id]
                    G.add_edge(qubit1, qubit2)
    return G, max_item


def local_contraction_sat_fac(G_fac, Ne_cavity, clauses, tendencies, cavity, node, max_item, egdelist2, device):
    if node > max_item:
        tensors = []
        eqin = ''
        eqout = ''
        node_nei = list(G_fac.neighbors(node))
        out_legs = sorted([item for item in node_nei if item in Ne_cavity])
        for out_leg in out_legs:
            eqout += oe.get_symbol(out_leg)
    else:
        tensors = [tc.tensor([0.5, 0.5]).to(device)]
        eqin = ''
        eqin += oe.get_symbol(node)
        eqin += ','
        eqout = ''
        eqout += oe.get_symbol(node)

    for noden in Ne_cavity:
        if noden > max_item:
            clause = clauses[noden - max_item - 1]
            tensor = (tc.ones(size=[2] * len(clause)) * (1.0)).to(device)
            zero_pos = [[int(-0.5 * i + 0.5)] for i in tendencies[noden - max_item - 1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            nodes_subset = sorted([item for item in clause if item not in Ne_cavity and item != node])
            nodes_subset_copy = copy.deepcopy(nodes_subset)
            for node_in_clause in clause:
                if node_in_clause in nodes_subset_copy:
                    Q = set(G_fac.neighbors(node_in_clause))
                    R = set(Ne_cavity)
                    if len(Q.intersection(R)) > 1:
                        symbol_id = egdelist2.index([node_in_clause, noden])
                        eqin += oe.get_symbol(symbol_id + max_item + 1)
                        node_in_clause_id = nodes_subset.index(node_in_clause)
                        nodes_subset[node_in_clause_id] = symbol_id + max_item + 1
                    else:
                        eqin += oe.get_symbol(node_in_clause)
                else:
                    eqin += oe.get_symbol(node_in_clause)
            eqin += ","
            if len(nodes_subset) == 3:
                tensor_pre = cavity[noden][node]
                tensor = tensor_pre[:8].reshape(2, 2, 2).to(device)
                tensors.append(tensor)
                for nodet in nodes_subset:
                    eqin += oe.get_symbol(nodet)
                eqin += ','
            if len(nodes_subset) == 2:
                tensor_pre = cavity[noden][node]
                tensor = tensor_pre[:4].reshape(2, 2).to(device)
                tensors.append(tensor)
                for nodet in nodes_subset:
                    eqin += oe.get_symbol(nodet)
                eqin += ','
            if len(nodes_subset) == 1:
                tensor = tensor_pre[:2].to(device)
                tensors.append(tensor)
                eqin += oe.get_symbol(nodes_subset[0])
                eqin += ","
        else:
            tensor = cavity[noden][node][:2].to(device)
            tensors.append(tensor)
            eqin += oe.get_symbol(noden)
            eqin += ","
    eqin = eqin.rstrip(',')
    eqin += '->'
    eq = eqin + eqout
    z = oe.contract(eq, *tensors, optimize=True)
    z = z.flatten()
    if z.numel() == 2:
        z = tc.cat([z, tc.zeros(6, device=device)])
    elif z.numel() != 8:
        if z.numel() < 8:
            z = tc.cat([z, tc.zeros(8 - z.numel(), device=device)])
        else:
            z = z[:8]
    if tc.equal(z, tc.zeros(8, device=device)):
        z = tc.ones(8, device=device) * 0.5
    return z


def local_contraction(Ne_cavity, clauses, tendencies, cavity, node, max_item, egdelist2, my_contract, device):
    if node > max_item:
        tensors = []
    else:
        tensors = [tc.tensor([0.5, 0.5]).to(device)]
    for noden in Ne_cavity:
        if noden > max_item:
            clause = clauses[noden - max_item - 1]
            tensor = (tc.ones(size=[2] * len(clause)) * (1.0)).to(device)
            zero_pos = [[int(-0.5 * i + 0.5)] for i in tendencies[noden - max_item - 1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            nodes_subset = sorted([item for item in clause if item not in Ne_cavity and item != node])
            if len(nodes_subset) == 3:
                tensor_pre = cavity[noden][node]
                tensor = tensor_pre[:8].reshape(2, 2, 2).to(device)
                tensors.append(tensor)
            if len(nodes_subset) == 2:
                tensor_pre = cavity[noden][node]
                tensor = tensor_pre[:4].reshape(2, 2).to(device)
                tensors.append(tensor)
            if len(nodes_subset) == 1:
                tensor = cavity[noden][node][:2].to(device)
                tensors.append(tensor)
        else:
            tensor = cavity[noden][node][:2].to(device)
            tensors.append(tensor)
    z = my_contract(*tensors)
    z = z.flatten()
    if z.numel() == 2:
        z = tc.cat([z, tc.zeros(6, device=device)])
    elif z.numel() != 8:
        if z.numel() < 8:
            z = tc.cat([z, tc.zeros(8 - z.numel(), device=device)])
        else:
            z = z[:8]
    if tc.equal(z, tc.zeros(8, device=device)):
        z = tc.ones(8, device=device) * 0.5
    return z


def local_contraction_begin(G_fac, Ne_cavity, clauses, tendencies, cavity, node, max_item, egdelist2, device):
    path = []
    if node > max_item:
        tensors = []
        eqin = ''
        eqout = ''
        node_nei = list(G_fac.neighbors(node))
        out_legs = sorted([item for item in node_nei if item in Ne_cavity])
        for out_leg in out_legs:
            eqout += oe.get_symbol(out_leg)
    else:
        tensors = [tc.tensor([0.5, 0.5]).to(device)]
        eqin = ''
        eqin += oe.get_symbol(node)
        eqin += ','
        eqout = ''
        eqout += oe.get_symbol(node)
        path.append((2,))
    for noden in Ne_cavity:
        if noden > max_item:
            clause = clauses[noden - max_item - 1]
            tensor = (tc.ones(size=[2] * len(clause)) * (1.0)).to(device)
            zero_pos = [[int(-0.5 * i + 0.5)] for i in tendencies[noden - max_item - 1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            path.append((2,) * len(clause))
            nodes_subset = sorted([item for item in clause if item not in Ne_cavity and item != node])
            nodes_subset_copy = copy.deepcopy(nodes_subset)
            for node_in_clause in clause:
                if node_in_clause in nodes_subset_copy:
                    Q = set(G_fac.neighbors(node_in_clause))
                    R = set(Ne_cavity)
                    if len(Q.intersection(R)) > 1:
                        symbol_id = egdelist2.index([node_in_clause, noden])
                        eqin += oe.get_symbol(symbol_id + max_item + 1)
                        node_in_clause_id = nodes_subset.index(node_in_clause)
                        nodes_subset[node_in_clause_id] = symbol_id + max_item + 1
                    else:
                        eqin += oe.get_symbol(node_in_clause)
                else:
                    eqin += oe.get_symbol(node_in_clause)
            eqin += ","
            if len(nodes_subset) == 3:
                tensor_pre = cavity[noden][node]
                tensor = tensor_pre[:8].reshape(2, 2, 2).to(device)
                tensors.append(tensor)
                for nodet in nodes_subset:
                    eqin += oe.get_symbol(nodet)
                eqin += ','
                path.append((2, 2, 2))
            if len(nodes_subset) == 2:
                tensor_pre = cavity[noden][node]
                tensor = tensor_pre[:4].reshape(2, 2).to(device)
                tensors.append(tensor)
                for nodet in nodes_subset:
                    eqin += oe.get_symbol(nodet)
                eqin += ','
                path.append((2, 2))
            if len(nodes_subset) == 1:
                tensor = cavity[noden][node][:2].to(device)
                tensors.append(tensor)
                eqin += oe.get_symbol(nodes_subset[0])
                eqin += ","
                path.append((2,))
        else:
            tensor = cavity[noden][node][:2].to(device)
            tensors.append(tensor)
            eqin += oe.get_symbol(noden)
            eqin += ","
            path.append((2,))
    eqin = eqin.rstrip(',')
    eqin += '->'
    eq = eqin + eqout

    def my_con_numpy(*tensors_torch):
        tensors_np = [t.cpu().detach().numpy() for t in tensors_torch]
        result_np = oe.contract(eq, *tensors_np, optimize=True)
        result_torch = tc.from_numpy(result_np).to(device)
        return result_torch

    z = my_con_numpy(*tensors)

    z = z.flatten()
    if z.numel() == 2:
        z = tc.cat([z, tc.zeros(6, device=device)])
    elif z.numel() != 8:
        if z.numel() < 8:
            z = tc.cat([z, tc.zeros(8 - z.numel(), device=device)])
        else:
            z = z[:8]
    if tc.equal(z, tc.zeros(8, device=device)):
        z = tc.ones(8, device=device) * 0.5
    return z, my_con_numpy


def inner_node(node, region_id, region_bound):
    if node in region_bound[region_id]:
        return False
    else:
        return True


# ======================================================================
# 修改后的 TNBP_3_sat_interaction_new 函数（区域选择性更新版本，适配4‑SAT）
# ======================================================================
def TNBP_3_sat_interaction_new(G_fac, Nv, boundaries, clauses, tendencies, interactions, max_item, region_info,
                               region_bound, device, cavity2=None):
    converged = True
    n = G_fac.number_of_nodes()
    step_limit = 100
    epsilon = 1e-3
    damping_factor = 0
    egdelist2 = [sorted(edge) for edge in G_fac.edges()]
    cavity = (tc.ones(size=(n, n, 8)) / 2).to(device)

    if cavity2 is not None:
        cavity = cavity2.clone()   # 使用热启动初值

    paths = [[[] for _ in range(n)] for _ in range(n)]
    t1 = time.time()

    # ======= 第一次迭代：计算每个区域的差异得分（region_scores） =======
    region_scores = [0.0] * len(region_info)
    for step in range(1):   # 只做一次迭代
        for div_id, div in enumerate(region_info):
            inner_msg = {}
            for center_node in div:
                neighborhood = Nv[center_node]
                for node in neighborhood:
                    # 非边界节点设为均匀分布
                    if node not in boundaries[center_node]:
                        default_msg = tc.zeros(8).to(device)
                        default_msg[0] = 1.0
                        default_msg[1] = 1.0
                        default_msg = default_msg / tc.sum(default_msg)
                        cavity[node][center_node] = default_msg
                        continue

                    Ne_cavity = list(set(Nv[node]).difference(Nv[center_node]))
                    for interaction in interactions:
                        for adn in interaction:
                            if node == adn:
                                for adc in interaction:
                                    if adc in Nv[center_node] and adc != adn and adc not in Ne_cavity and adc != center_node:
                                        Ne_cavity.append(adc)
                    Ne_cavity = list(set(Ne_cavity))

                    if len(Ne_cavity) == 0:
                        continue

                    # 收缩并记录路径
                    New_cavity_vec, my_contract = local_contraction_begin(
                        G_fac, Ne_cavity, clauses, tendencies, cavity,
                        node, max_item, egdelist2, device
                    )
                    paths[node][center_node] = my_contract

                    # 计算差异（使用 L1 总和作为边差异）
                    diff = tc.sum(tc.abs(New_cavity_vec - cavity[node][center_node])).item()
                    region_scores[div_id] += diff   # 累加到当前区域得分

                    # 阻尼更新
                    temp = damping_factor * cavity[node][center_node] + (1 - damping_factor) * New_cavity_vec
                    temp /= tc.sum(temp)
                    cavity[node][center_node] = temp

                    # 内部消息缓存
                    is_inner = (center_node not in region_bound[div_id]) and (node in region_bound[div_id])
                    if is_inner:
                        inner_msg[node] = temp

        t2 = time.time()
        print(f"Initial iteration done. Region scores computed. time: {t2 - t1:.2f}s")

    # ======= 根据 region_scores 确定需要更新的“坏区域” =======
    if cavity2 is not None:
        # 按得分降序排序，取前 30% 的区域作为坏区域
        sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
        top_k = max(1, int(len(region_scores) * 0.3))
        bad_regions = set(sorted_indices[:top_k])
        print(f"Bad regions (top {top_k}/{len(region_scores)}): {bad_regions}")
    else:
        # 没有热启动时，全部区域都需要更新
        bad_regions = set(range(len(region_info)))
        print("No warm-start, updating all regions.")

    # ======= 后续迭代：只更新坏区域内的消息 =======
    for step in range(1, step_limit):
        diff_max = 0.0
        for div_id, div in enumerate(region_info):
            # 跳过非坏区域
            if cavity2 is not None and div_id not in bad_regions:
                continue

            inner_msg = {}
            for center_node in div:
                neighborhood = Nv[center_node]
                for node in neighborhood:
                    if node not in boundaries[center_node]:
                        default_msg = tc.zeros(8).to(device)
                        default_msg[0] = 1.0
                        default_msg[1] = 1.0
                        default_msg = default_msg / tc.sum(default_msg)
                        cavity[node][center_node] = default_msg
                        continue

                    Ne_cavity = list(set(Nv[node]).difference(Nv[center_node]))
                    for interaction in interactions:
                        for adn in interaction:
                            if node == adn:
                                for adc in interaction:
                                    if adc in Nv[center_node] and adc != adn and adc not in Ne_cavity and adc != center_node:
                                        Ne_cavity.append(adc)
                    Ne_cavity = list(set(Ne_cavity))

                    if len(Ne_cavity) == 0:
                        continue

                    # 使用已保存的收缩路径
                    if paths[node][center_node] == []:
                        New_cavity_vec, my_contract = local_contraction_begin(
                            G_fac, Ne_cavity, clauses, tendencies, cavity,
                            node, max_item, egdelist2, device
                        )
                        paths[node][center_node] = my_contract
                    else:
                        New_cavity_vec = local_contraction(
                            Ne_cavity, clauses, tendencies, cavity, node,
                            max_item, egdelist2, paths[node][center_node], device
                        )

                    temp = damping_factor * cavity[node][center_node] + (1 - damping_factor) * New_cavity_vec
                    temp /= tc.sum(temp)

                    # 计算最大分量差异（用于收敛判断）
                    diff = max(
                        abs(temp[0] - cavity[node][center_node][0]).item(),
                        abs(temp[1] - cavity[node][center_node][1]).item(),
                        abs(temp[2] - cavity[node][center_node][2]).item(),
                        abs(temp[3] - cavity[node][center_node][3]).item(),
                        abs(temp[4] - cavity[node][center_node][4]).item(),
                        abs(temp[5] - cavity[node][center_node][5]).item(),
                        abs(temp[6] - cavity[node][center_node][6]).item(),
                        abs(temp[7] - cavity[node][center_node][7]).item()
                    )
                    diff_max = max(diff_max, diff)

                    cavity[node][center_node] = temp

                    is_inner = (center_node not in region_bound[div_id]) and (node in region_bound[div_id])
                    if is_inner:
                        inner_msg[node] = temp

        t2 = time.time()
        print(f"iteration step: {step + 1}, difference: {diff_max:.6f}, time: {t2 - t1:.2f}s")
        if diff_max <= epsilon:
            break

    if step == step_limit - 1:
        print('unconverged')
        converged = False

    return cavity, converged, egdelist2, step + 1
# ======================================================================