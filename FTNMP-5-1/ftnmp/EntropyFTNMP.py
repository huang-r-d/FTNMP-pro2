import torch as tc
import opt_einsum as oe
import TNBP as TNBP
import nx_test as nxt
import get_regions_new as grn


def entropy_single_point(single_point, G_fac, clauses, tendencies, devides, devide_bound, cavity, egdelist2,
                         degree_node, max_item, device):
    tensors = []
    eq_parts = []  # 收集每个顶层张量的符号
    for bd_id in range(len(devide_bound)):
        bd = devide_bound[bd_id]
        if single_point in bd:
            devide = devides[bd_id]
            subtensors = []
            sub_parts = []  # 收集每个子张量的符号
            for node in devide:
                if node > max_item and node - max_item - 1 < len(clauses):
                    clause = clauses[node - max_item - 1]
                    subtensor = (tc.ones(size=[2] * len(clause)) * (1.0)).to(device)
                    zero_pos = [[int(-0.5 * i + 0.5)] for i in tendencies[node - max_item - 1]]
                    subtensor[zero_pos] = 0.0
                    subtensors.append(subtensor)
                    sym = ''
                    for node_in_clause in clause:
                        Q = set(G_fac.neighbors(node_in_clause))
                        R = set(devide)
                        if len(Q.intersection(R)) > 1 and node in bd and node_in_clause not in devide:
                            symbol_id = egdelist2.index([node_in_clause, node])
                            sym += oe.get_symbol(symbol_id + max_item + 1)
                        else:
                            sym += oe.get_symbol(node_in_clause)
                    sub_parts.append(sym)
                if G_fac.degree(node) == 1:
                    subtensor = (tc.tensor([0.5, 0.5])).to(device)
                    subtensors.append(subtensor)
                    sub_parts.append(oe.get_symbol(node))

            for nodey in [i for i in bd if i != single_point]:
                nodey_nei = list(G_fac.neighbors(nodey))
                message = sorted([item for item in nodey_nei if item not in devide], reverse=False)
                Nv_pre = [(n, len(devide)) for n in list(set(nodey_nei).difference(set(message)))]
                max_tuple = max(Nv_pre, key=lambda x: x[1])
                cavity_id = devide.index(max_tuple[0])
                tensor_pre = cavity[nodey][devide[cavity_id]]

                if nodey > max_item:  # 因子节点
                    if len(message) > 0:
                        sz = 2 ** len(message)
                        shape = [2] * len(message)
                        subtensor = tensor_pre[:sz].reshape(shape).to(device)
                        subtensors.append(subtensor)
                        sym = ''
                        for mid in range(len(message)):
                            Q = set(G_fac.neighbors(message[mid]))
                            R = set(devide)
                            if len(Q.intersection(R)) > 1:
                                symbol_id = egdelist2.index([message[mid], nodey])
                                sym += oe.get_symbol(symbol_id + max_item + 1)
                            else:
                                sym += oe.get_symbol(message[mid])
                        sub_parts.append(sym)
                    else:
                        # 无外部腿，收缩为标量
                        subtensor = tc.sum(tensor_pre).to(device)
                        subtensors.append(subtensor)
                        sub_parts.append('')  # 空字符串表示标量，在join时会被保留位置
                else:  # 变量节点
                    subtensor = tensor_pre[:2].clone().to(device)
                    subtensors.append(subtensor)
                    sub_parts.append(oe.get_symbol(nodey))

            subeqin = ",".join(sub_parts)
            subeqin += "->" + oe.get_symbol(single_point)
            subz = oe.contract(subeqin, *subtensors, optimize=True)
            subz /= tc.sum(subz)
            tensors.append(subz)
            eq_parts.append(oe.get_symbol(single_point))

    eq = ",".join(eq_parts) + "->"
    z = oe.contract(eq, *tensors)
    z = ((degree_node[single_point] - 1)) * tc.log(z)
    z_errors = []
    for i in range(len(tensors)):
        tensors_error = [tensors[j] for j in range(len(tensors)) if j != i]
        if len(tensors_error) > 0:
            eq_error = ",".join(['a'] * len(tensors_error)) + "->"
            z_error = oe.contract(eq_error, *tensors_error, optimize=True)
            z_errors.append(z_error)
    Error = z - tc.log(tc.prod(tc.tensor(z_errors)))
    return Error


def entropy_cluster(tar_node, G_fac, Nv, boundaries, clauses, tendencies, cavity, max_item, nei_cxx, device):
    egdelist2 = [sorted(edge) for edge in G_fac.edges()]
    tensors = []
    eq_parts = []
    for noden in Nv[tar_node]:
        if noden > max_item and noden - max_item - 1 < len(clauses):
            clause = clauses[noden - max_item - 1]
            tensor = (tc.ones(size=[2] * len(clause)) * (1.0)).to(device)
            zero_pos = [[int(-0.5 * i + 0.5)] for i in tendencies[noden - max_item - 1]]
            tensor[zero_pos] = 0.0
            tensors.append(tensor)
            sym = ''
            for node_in_clause in clause:
                Q = set(G_fac.neighbors(node_in_clause))
                R = set(Nv[tar_node])
                if len(Q.intersection(R)) > 1 and noden in boundaries[tar_node] and node_in_clause not in Nv[tar_node]:
                    symbol_id = egdelist2.index([node_in_clause, noden])
                    sym += oe.get_symbol(symbol_id + max_item + 1)
                else:
                    sym += oe.get_symbol(node_in_clause)
            eq_parts.append(sym)
        if len(Nv[tar_node]) != 2 and G_fac.degree(noden) == 1 and noden not in boundaries[tar_node]:
            tensor = (tc.tensor([0.5, 0.5])).to(device)
            tensors.append(tensor)
            eq_parts.append(oe.get_symbol(noden))

    for nodey in boundaries[tar_node]:
        nodey_nei = list(G_fac.neighbors(nodey))
        message = sorted([item for item in nodey_nei if item not in Nv[tar_node]], reverse=False)
        Nv_pre = [(n, len(nei_cxx[n])) for n in list(set(nodey_nei).difference(set(message)))]
        max_tuple = max(Nv_pre, key=lambda x: x[1])
        cavity_id = Nv[tar_node].index(max_tuple[0])
        tensor_pre = cavity[nodey][Nv[tar_node][cavity_id]]

        if nodey > max_item:  # 因子节点
            if len(message) > 0:
                sz = 2 ** len(message)
                shape = [2] * len(message)
                tensor = tensor_pre[:sz].reshape(shape).to(device)
                tensors.append(tensor)
                sym = ''
                for mid in range(len(message)):
                    Q = set(G_fac.neighbors(message[mid]))
                    R = set(Nv[tar_node])
                    if len(Q.intersection(R)) > 1:
                        symbol_id = egdelist2.index([message[mid], nodey])
                        sym += oe.get_symbol(symbol_id + max_item + 1)
                    else:
                        sym += oe.get_symbol(message[mid])
                eq_parts.append(sym)
            else:
                tensor = tc.sum(tensor_pre).to(device)
                tensors.append(tensor)
                eq_parts.append('')
        else:  # 变量节点
            tensor = tensor_pre[:2].clone().to(device)
            tensors.append(tensor)
            eq_parts.append(oe.get_symbol(nodey))

    eq = ",".join(eq_parts) + "->"
    if len(tensors) != 0:
        marginal = oe.contract(eq, *tensors, optimize=True)
    else:
        marginal = (tc.tensor(2)).to(device)
    return marginal


def entropies_cluster(G_fac, clauses, tendencies, devide, devide_bound, max_item, nei_cxx, cavity, device):
    marginal_devides = []
    for devide_id in range(len(devide)):
        marginal_devide = entropy_cluster(devide_id, G_fac, devide, devide_bound, clauses, tendencies, cavity, max_item,
                                          nei_cxx, device)
        marginal_devides.append(tc.log(marginal_devide))
    return marginal_devides


def entropies_point(single_list, G_fac, clauses, tendencies, devides, devide_bound, cavity, edgelist2, degree_node,
                    max_item, device):
    logs_point = []
    for node in set(single_list):
        if node in degree_node:
            m = entropy_single_point(node, G_fac, clauses, tendencies, devides, devide_bound, cavity, edgelist2,
                                     degree_node, max_item, device)
            logs_point.append(m)
    return logs_point


def entropy_intersection(G_fac, edges, clauses, tendencies, cavity, max_item, Nv, device):
    ms = []
    for edge_id in range(len(edges)):
        m = entropy_cluster(edge_id, G_fac, edges, edges, clauses, tendencies, cavity, max_item, Nv, device)
        ms.append(tc.log(m))
    return ms


def EntropyZ_new(G_fac, clauses, tendencies, devides, devide_bound, region_info, single_list, max_item, Nv, boundaries,
                 degree_node, device, file, cavity2=None):
    intersections_info, edges_info, direct_products_info, else_situations = nxt.cluster_edges_generate_new(G_fac,
                                                                                                           max_item,
                                                                                                           devides,
                                                                                                           devide_bound)
    direct_products = [sorted(direct_products_info[i][2]) for i in range(len(direct_products_info)) if
                       len(direct_products_info[i][2]) >= 2 and min(direct_products_info[i][0],
                                                                    direct_products_info[i][1]) <= len(region_info)]
    region_info, region_bound = grn.region_info_change(G_fac, clauses, region_info, max_item, single_list)
    cavity, converged, edgelist2, step = TNBP.TNBP_3_sat_interaction_new(G_fac, Nv, boundaries, clauses, tendencies,
                                                                         direct_products, max_item, region_info,
                                                                         region_bound, device, cavity2)
    intersections = [sorted(edges_info[i][2]) for i in range(len(edges_info))]
    loges_devides = entropies_cluster(G_fac, clauses, tendencies, devides, devide_bound, max_item, Nv, cavity, device)
    logs_points = entropies_point(single_list, G_fac, clauses, tendencies, devides, devide_bound, cavity, edgelist2,
                                  degree_node, max_item, device)
    ms = sum(entropy_intersection(G_fac, intersections, clauses, tendencies, cavity, max_item, Nv, device))
    entropy = sum(loges_devides) - sum(logs_points) - ms

    return entropy, step, cavity