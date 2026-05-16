import torch as tc
import networkx as nx
import copy 
import opt_einsum as oe
import time
import numpy as np

def G_generator_fac(clauses):
    """
    Generate a graph based on the given list of clauses and compute the maximum node identifier.

    Parameters:
    clauses (list of lists): Each clause is a list of integers representing nodes connected by edges in the graph. Nodes within a clause are connected to an additional node representing the clause.

    Returns:
    G (nx.Graph): The generated undirected graph where nodes and edges are defined by the clauses.
    max_item (int): The highest node identifier in the graph, which is the maximum variable node index across all clauses plus the number of clauses.

    """
    G = nx.Graph()    
    if len(clauses)==0:
        max_item = 0
    else:
        max_item = max(max(row) for row in clauses if len(row)!=0)
        for clause_id in range(len(clauses)):
            clause = clauses[clause_id]
            if len(clause)!=0:
                qubit2 = max_item + clause_id + 1
                for qubit1_id in range(len(clause)):
                    qubit1 = clause[qubit1_id]
                    G.add_edge(qubit1,qubit2)
    return G,max_item

def local_contraction_sat_fac(G_fac,Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,device):
    """
    Perform local tensor contraction for a specific node in a graph, esing tensors and equations based on the node's cavity sungraph and clauses.

    Parameters:
    G_fac (nx.Graph): The factor graph representing the problem.
    Ne_cavity (list of int): List of nodes in the cavity.
    clauses (list of lists): List of clauses, where each clause is a list of integers representing nodes.
    tendencies (list of lists): List of tendencies for nodes, indicating which tensor positions should be set to zero.
    cavity (list of dicts): List of dictionaries containing tensors for the cavity nodes.
    node (int): The current node to perform contraction on.
    max_item (int): The maximum node identifier used in the graph.
    egdelist2 (list of lists): List of edges in the graph, where each edge is represented as a list of two nodes.
    device (torch.device): The device (CPU or GPU) on which tensors are allocated.

    Returns:
    z (torch.Tensor): The result of the tensor contraction, adjusted for cases where the resulting tensor is zero.
    """
    if node>max_item:
        tensors = []
        eqin = ''
        eqout = ''
        node_nei = list(G_fac.neighbors(node))
        out_legs = sorted([item for item in node_nei if item in Ne_cavity])
        for out_leg in out_legs:
            eqout += oe.get_symbol(out_leg)
    else:
        tensors = [tc.tensor([0.5,0.5]).to(device)]
        eqin = ''
        eqin += oe.get_symbol(node)
        eqin +=','
        eqout = ''
        eqout += oe.get_symbol(node)

    for noden in Ne_cavity:
        if noden>max_item:
    
            clause = clauses[noden - max_item-1]
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            nodes_subset =sorted([item for item in clause if item not in Ne_cavity and item != node])
            nodes_subset_copy = copy.deepcopy(nodes_subset)
            for node_in_clause in clause:
                if node_in_clause  in nodes_subset_copy:
                    Q = set(G_fac.neighbors(node_in_clause))
                    R = set(Ne_cavity)
                    if len(Q.intersection(R))>1:
                        symbol_id = egdelist2.index([node_in_clause,noden])
                        eqin +=oe.get_symbol(symbol_id+max_item+1)
                        node_in_clause_id = nodes_subset.index(node_in_clause)
                        nodes_subset[node_in_clause_id]=symbol_id+max_item+1
                    else:
                        eqin += oe.get_symbol(node_in_clause)
                else:
                    eqin += oe.get_symbol(node_in_clause)
            eqin += ","
            if len(nodes_subset)==2:
                tensor_pre = cavity[noden][node]
                tensor=(tensor_pre.reshape(2,2)).to(device)
                tensors.append(tensor)
                for nodet in nodes_subset:
                    eqin +=oe.get_symbol(nodet)
                eqin +=','
            if len(nodes_subset)==1:
                tensor = (tc.tensor([cavity[noden][node][0],cavity[noden][node][1]])).to(device)
                tensors.append(tensor)
                eqin += oe.get_symbol(nodes_subset[0])
                eqin += ","
        else:
            tensor = tc.tensor([cavity[noden][node][0],cavity[noden][node][1]]).to(device)
            tensors.append(tensor)
            eqin += oe.get_symbol(noden)
            eqin += ","
    eqin = eqin.rstrip(',')
    eqin += '->'
    eq = eqin + eqout
    z = oe.contract(eq, *tensors, optimize=True)
    if len(z.size())== 1:
       z = (tc.tensor([z[0],z[1],0,0])).to(device)
    if len(z.size())== 2:
       z=(z.reshape(4)).to(device) 
    con = (tc.zeros(4)).to(device)
    if tc.equal(z,con)==True:
        z =(tc.tensor([0.5,0.5,0.5,0.5])).to(device)
    return z

def local_contraction(Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,my_contract,device):
    """
    Perform local tensor contraction for a specific node in a graph using a provided contraction function.

    Parameters:
    Ne_cavity (list of int): List of nodes in the cavity.
    clauses (list of lists): List of clauses, where each clause is a list of integers representing nodes.
    tendencies (list of lists): List of tendencies for nodes, indicating which tensor positions should be set to zero.
    cavity (list of dicts): List of dictionaries containing tensors for the cavity nodes.
    node (int): The current node to perform contraction on.
    max_item (int): The maximum node identifier used in the graph.
    egdelist2 (list of lists): List of edges in the graph, where each edge is represented as a list of two nodes (not used in this function).
    my_contract (function): Function to perform the tensor contraction.
    device (torch.device): The device (CPU or GPU) on which tensors are allocated.

    Returns:
    z (torch.Tensor): The result of the tensor contraction, adjusted for cases where the resulting tensor is zero.
    """
    if node>max_item:
        tensors = []
    else:
        tensors = [tc.tensor([0.5,0.5]).to(device)]
    for noden in Ne_cavity:
        if noden>max_item:
            clause = clauses[noden - max_item-1]
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            nodes_subset =sorted([item for item in clause if item not in Ne_cavity and item != node])
            if len(nodes_subset)==2:
                tensor_pre = cavity[noden][node]
                tensor=(tensor_pre.reshape(2,2)).to(device)
                tensors.append(tensor)
            if len(nodes_subset)==1:
                tensor = (tc.tensor([cavity[noden][node][0],cavity[noden][node][1]])).to(device)
                tensors.append(tensor)
        else:
            tensor = tc.tensor([cavity[noden][node][0],cavity[noden][node][1]]).to(device)
            tensors.append(tensor)
    z = my_contract(*tensors)
    if len(z.size())== 1:
       z = (tc.tensor([z[0],z[1],0,0])).to(device)
    if len(z.size())== 2:
       z=(z.reshape(4)).to(device) 
    con = (tc.zeros(4)).to(device)
    if tc.equal(z,con)==True:
        z =(tc.tensor([0.5,0.5,0.5,0.5])).to(device)
    return z

def local_contraction_begin(G_fac,Ne_cavity,clauses,tendencies,cavity,node,max_item,egdelist2,device):
    """
    Perform an initial local tensor contraction for a specific node in a graph and prepare a contraction path for future use.

    Parameters:
    G_fac (nx.Graph): The factor graph representing the problem.
    Ne_cavity (list of int): List of nodes in the cavity.
    clauses (list of lists): List of clauses, where each clause is a list of integers representing nodes.
    tendencies (list of lists): List of tendencies for nodes, indicating which tensor positions should be set to zero.
    cavity (list of dicts): List of dictionaries containing tensors for the cavity nodes.
    node (int): The current node to perform contraction on.
    max_item (int): The maximum node identifier used in the graph.
    egdelist2 (list of lists): List of edges in the graph, where each edge is represented as a list of two nodes.
    device (torch.device): The device (CPU or GPU) on which tensors are allocated.

    Returns:
    z (torch.Tensor): The result of the tensor contraction, adjusted for cases where the resulting tensor is zero.
    my_con (function): A function for performing the contraction path.
    """
    path = []
    if node>max_item:
        tensors = []
        eqin = ''
        eqout = ''
        node_nei = list(G_fac.neighbors(node))
        out_legs = sorted([item for item in node_nei if item in Ne_cavity])
        for out_leg in out_legs:
            eqout += oe.get_symbol(out_leg)
    else:
        tensors = [tc.tensor([0.5,0.5]).to(device)]
        eqin = ''
        eqin += oe.get_symbol(node)
        eqin +=','
        eqout = ''
        eqout += oe.get_symbol(node)
        path.append((2,))
    for noden in Ne_cavity:
        if noden>max_item:
            clause = clauses[noden - max_item-1]
            tensor = (tc.ones(size=[2]*len(clause))*(1.0)).to(device)
            zero_pos = [[int(-0.5*i+0.5)] for i in tendencies[noden - max_item-1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            path.append((2,)*len(clause))
            nodes_subset =sorted([item for item in clause if item not in Ne_cavity and item != node])
            nodes_subset_copy = copy.deepcopy(nodes_subset)
            for node_in_clause in clause:
                if node_in_clause  in nodes_subset_copy:
                    Q = set(G_fac.neighbors(node_in_clause))
                    R = set(Ne_cavity)
                    if len(Q.intersection(R))>1:
                        symbol_id = egdelist2.index([node_in_clause,noden])
                        eqin +=oe.get_symbol(symbol_id+max_item+1)
                        node_in_clause_id = nodes_subset.index(node_in_clause)
                        nodes_subset[node_in_clause_id]=symbol_id+max_item+1
                    else:
                        eqin += oe.get_symbol(node_in_clause)
                else:
                    eqin += oe.get_symbol(node_in_clause)
            eqin += ","
            if len(nodes_subset)==2:
                tensor_pre = cavity[noden][node]
                tensor=(tensor_pre.reshape(2,2)).to(device)
                tensors.append(tensor)
                for nodet in nodes_subset:
                    eqin +=oe.get_symbol(nodet)
                eqin +=','
                path.append((2,2))
            if len(nodes_subset)==1:
                tensor = (tc.tensor([cavity[noden][node][0],cavity[noden][node][1]])).to(device)
                tensors.append(tensor)
                eqin += oe.get_symbol(nodes_subset[0])
                eqin += ","
                path.append((2,))
        else:
            tensor = tc.tensor([cavity[noden][node][0],cavity[noden][node][1]]).to(device)
            tensors.append(tensor)
            eqin += oe.get_symbol(noden)
            eqin += ","
            path.append((2,))
    eqin = eqin.rstrip(',')
    eqin += '->'
    eq = eqin + eqout

    # 定义使用 NumPy 的收缩函数（避免 PyTorch 的 25 维限制）
    def my_con_numpy(*tensors_torch):
        # 将所有张量转为 NumPy 数组（移至 CPU）
        tensors_np = [t.cpu().detach().numpy() for t in tensors_torch]
        # 使用 opt_einsum.contract 进行收缩，后端自动选择（会使用 numpy）
        result_np = oe.contract(eq, *tensors_np, optimize=True)
        # 转回 PyTorch 张量，并放回原设备
        result_torch = tc.from_numpy(result_np).to(device)
        return result_torch

    # 第一次执行收缩
    z = my_con_numpy(*tensors)

    if len(z.size()) == 1:
        z = (tc.tensor([z[0], z[1], 0, 0])).to(device)
    if len(z.size()) == 2:
        z = (z.reshape(4)).to(device)
    con = (tc.zeros(4)).to(device)
    if tc.equal(z, con) == True:
        z = (tc.tensor([0.5, 0.5, 0.5, 0.5])).to(device)
    return z, my_con_numpy  # 注意返回的是 my_con_numpy


def inner_node(node, region_id,region_bound):
    """
    Determine if a node is an inner node within a specific region.

    Parameters:
    node (int): The node identifier to check.
    region_id (int): The identifier of the region in which to check the node.
    region_bound (list of lists): List of boundaries, where each region is a list of integers representing nodes in its boundary.

    Returns:
    bool: `True` if the node is not in the boundary of the specified region, indicating it is an inner node. `False` otherwise.
    """
    if node in region_bound[region_id]:
        return False
    else:
        return True

def TNBP_3_sat_interaction_new(G_fac, Nv, boundaries, clauses, tendencies, interactions,
                               max_item, region_info, region_bound, device, cavity2=None):
    """
    Tensor Network Belief Propagation with region-level skipping.
    Only updates regions with large cavity changes in the first iteration.
    """
    converged = True
    n = G_fac.number_of_nodes()
    step_limit = 20
    epsilon = 1e-3
    damping_factor = 0.0
    egdelist2 = [sorted(edge) for edge in G_fac.edges()]
    cavity = (tc.ones(size=(n, n, 4)) / 2).to(device)
    if cavity2 is not None:
        cavity = cavity2.clone()   # 使用传入的初始腔场

    paths = [[[] for _ in range(n)] for _ in range(n)]
    t1 = time.time()

    # ---------- 第一次迭代：计算每个区域的差异总分 ----------
    region_scores = [0.0] * len(region_info)   # 每个区域的差异总分
    # 记录第一次迭代中每个区域需要更新的消息（可选，用于后续跳过）
    for step in range(1):   # 只做一次迭代
        for div_id, div in enumerate(region_info):
            inner_msg = {}
            for center_node in div:
                neighborhood = Nv[center_node]
                for node in neighborhood:
                    # 仅处理边界节点
                    if node not in boundaries[center_node]:
                        # 非边界节点直接设为均匀分布（原逻辑）
                        cavity[node][center_node] = tc.tensor([1.0, 1.0, 0, 0]).to(device)
                        cavity[node][center_node] /= tc.sum(cavity[node][center_node])
                        continue

                    # 计算 Ne_cavity
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

                    # 执行收缩（第一次迭代，必须初始化路径）
                    New_cavity_vec, my_contract = local_contraction_begin(
                        G_fac, Ne_cavity, clauses, tendencies, cavity,
                        node, max_item, egdelist2, device
                    )
                    paths[node][center_node] = my_contract

                    # 计算差异（L1范数）
                    diff = tc.sum(tc.abs(New_cavity_vec - cavity[node][center_node])).item()
                    # 累加到当前区域的得分
                    region_scores[div_id] += diff

                    # 更新消息（带阻尼）
                    temp = damping_factor * cavity[node][center_node] + (1 - damping_factor) * New_cavity_vec
                    temp /= tc.sum(temp)
                    cavity[node][center_node] = temp

                    # 内部消息缓存
                    is_inner = (center_node not in region_bound[div_id]) and (node in region_bound[div_id])
                    if is_inner:
                        inner_msg[node] = temp

        # 第一次迭代结束，不再继续（后续迭代将在下面独立循环）
        print(f"Initial iteration done. Region scores computed.")

    # ---------- 确定需要更新的“坏区域” ----------
    if cavity2 is not None:
        # 按得分降序排序，取前 30% 的区域作为坏区域
        sorted_indices = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
        top_k = max(1, int(len(region_scores) * 0.3))   # 可调整比例
        bad_regions = set(sorted_indices[:top_k])
        print(f"Bad regions (top {top_k}/{len(region_scores)}): {bad_regions}")
    else:
        bad_regions = set(range(len(region_info)))   # 如果没有 warm‑start，全部更新

    # ---------- 后续迭代（只更新坏区域） ----------
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
                        # 非边界节点：设均匀分布（始终更新，因为不依赖外部消息）
                        cavity[node][center_node] = tc.tensor([1.0, 1.0, 0, 0]).to(device)
                        cavity[node][center_node] /= tc.sum(cavity[node][center_node])
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

                    # 使用已保存的 contraction 路径
                    if paths[node][center_node] == []:
                        # 防御：如果路径未初始化，重新初始化
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

                    # 阻尼更新
                    temp = damping_factor * cavity[node][center_node] + (1 - damping_factor) * New_cavity_vec
                    temp /= tc.sum(temp)

                    # 计算最大分量差异
                    diff = max(
                        abs(temp[0] - cavity[node][center_node][0]).item(),
                        abs(temp[1] - cavity[node][center_node][1]).item(),
                        abs(temp[2] - cavity[node][center_node][2]).item(),
                        abs(temp[3] - cavity[node][center_node][3]).item()
                    )
                    diff_max = max(diff_max, diff)

                    cavity[node][center_node] = temp
                    is_inner = (center_node not in region_bound[div_id]) and (node in region_bound[div_id])
                    if is_inner:
                        inner_msg[node] = temp

        t2 = time.time()
        print(f"iteration step: {step+1}, difference: {diff_max:.6f}, time: {t2-t1:.2f}s")
        if diff_max <= epsilon:
            break

    if step == step_limit - 1:
        print('unconverged')
        converged = False

    return cavity, converged, egdelist2, step + 1
