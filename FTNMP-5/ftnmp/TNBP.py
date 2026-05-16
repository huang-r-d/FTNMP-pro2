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


def TNBP_3_sat_interaction_new(G_fac, Nv, boundaries, clauses, tendencies, interactions, max_item, region_info,
                               region_bound, device, cavity2=None):
    converged = True
    n = G_fac.number_of_nodes()
    step_limit = 100
    epsilon = 1e-3
    difference_max = tc.tensor([0.0]).to(device)
    damping_factor = 0
    egdelist2 = [sorted(edge) for edge in G_fac.edges()]
    cavity = (tc.ones(size=(n, n, 8)) / 2).to(device)

    if cavity2 is not None:
        cavity = cavity2
        bad_cavity_points = set()

    paths = [[[] for w in range(n)] for d in range(n)]
    t1 = time.time()
    tt = 0
    for step in range(0, 1):
        first_center_node = -1
        for div_id in range(len(region_info)):
            div = region_info[div_id]
            inner_msg = {}
            for center_node in div:
                first_boundary_node = -1
                neighborhood = Nv[center_node]
                if len(boundaries[center_node]) != 0 and first_center_node == -1:
                    first_center_node = center_node
                for node in neighborhood:
                    is_inner_msg = inner_node(center_node, div_id, region_bound) and not inner_node(node, div_id, region_bound)
                    if (node in inner_msg) and is_inner_msg:
                        cavity[node][center_node] = inner_msg[node]
                        continue
                    if node in boundaries[center_node]:
                        if first_boundary_node == -1:
                            first_boundary_node = node
                        Ne_cavity = list(set(Nv[node]).difference(Nv[center_node]))
                        for interaction in interactions:
                            for adn in interaction:
                                if node == adn:
                                    for adc in interaction:
                                        if adc in Nv[center_node] and adc != adn and adc not in Ne_cavity and adc != center_node:
                                            Ne_cavity.append(adc)
                                            Ne_cavity = list(set(Ne_cavity))
                        if len(Ne_cavity) != 0:
                            if step > 0:
                                New_cavity_vec = local_contraction(Ne_cavity, clauses, tendencies, cavity, node,
                                                                   max_item, egdelist2, paths[node][center_node], device)
                            else:
                                New_cavity_vec, my_contract = local_contraction_begin(G_fac, Ne_cavity, clauses,
                                                                                      tendencies, cavity, node,
                                                                                      max_item, egdelist2, device)
                                paths[node][center_node] = my_contract
                                if cavity2 is not None:
                                    bad_threshold = 0.01
                                    for noden in Ne_cavity:
                                        diff = tc.abs(New_cavity_vec - cavity[noden][node])
                                        if diff.max() > bad_threshold:
                                            bad_cavity_points.add((noden, node))
                            temp = damping_factor * cavity[node][center_node] + (1 - damping_factor) * New_cavity_vec
                            temp /= tc.sum(temp)
                            difference = max(abs(temp - cavity[node][center_node]))
                            cavity[node][center_node] = temp
                            if is_inner_msg:
                                inner_msg[node] = temp
                        else:
                            difference = tc.tensor(0.0).to(device)
                    else:
                        default_msg = tc.zeros(8).to(device)
                        default_msg[0] = 1.0
                        default_msg[1] = 1.0
                        default_msg = default_msg / tc.sum(default_msg)
                        cavity[node][center_node] = default_msg
                        difference = tc.tensor(0.0).to(device)
                    if center_node == first_center_node and node == first_boundary_node:
                        difference_max = difference
                    elif difference > difference_max:
                        difference_max = difference
        t2 = time.time()
        tt = difference_max.item()
        print("iteration step:", step + 1, ",   difference:", difference_max.item(), '    time:', t2 - t1)
        if difference_max <= epsilon:
            break

    skip = 0
    skip2 = 0
    noskip = 0
    noskip2 = 0
    flag = True

    for step in range(1, step_limit):
        first_center_node = -1
        for div_id in range(len(region_info)):
            div = region_info[div_id]
            inner_msg = {}
            if cavity2 is not None and flag:
                region_has_bad = False
                for center_node in div:
                    for noden in Nv[center_node]:
                        if (noden, center_node) in bad_cavity_points:
                            region_has_bad = True
                            break
                    if region_has_bad:
                        break
                if not region_has_bad:
                    skip += 1
                    continue
                else:
                    noskip += 1
            for center_node in div:
                first_boundary_node = -1
                neighborhood = Nv[center_node]
                if len(boundaries[center_node]) != 0 and first_center_node == -1:
                    first_center_node = center_node
                for node in neighborhood:
                    if cavity2 is not None and (node, center_node) not in bad_cavity_points and flag:
                        skip2 += 1
                        continue
                    else:
                        noskip2 += 1
                    is_inner_msg = inner_node(center_node, div_id, region_bound) and not inner_node(node, div_id, region_bound)
                    if (node in inner_msg) and is_inner_msg:
                        cavity[node][center_node] = inner_msg[node]
                        continue
                    if node in boundaries[center_node]:
                        if first_boundary_node == -1:
                            first_boundary_node = node
                        Ne_cavity = list(set(Nv[node]).difference(Nv[center_node]))
                        for interaction in interactions:
                            for adn in interaction:
                                if node == adn:
                                    for adc in interaction:
                                        if adc in Nv[center_node] and adc != adn and adc not in Ne_cavity and adc != center_node:
                                            Ne_cavity.append(adc)
                                            Ne_cavity = list(set(Ne_cavity))
                        if len(Ne_cavity) != 0:
                            if step > 0:
                                if paths[node][center_node] == []:
                                    New_cavity_vec, my_contract = local_contraction_begin(
                                        G_fac, Ne_cavity, clauses, tendencies, cavity,
                                        node, max_item, egdelist2, device)
                                    paths[node][center_node] = my_contract
                                else:
                                    New_cavity_vec = local_contraction(
                                        Ne_cavity, clauses, tendencies, cavity, node,
                                        max_item, egdelist2, paths[node][center_node], device)
                            else:
                                New_cavity_vec, my_contract = local_contraction_begin(G_fac, Ne_cavity, clauses,
                                                                                      tendencies, cavity, node,
                                                                                      max_item, egdelist2, device)
                                paths[node][center_node] = my_contract
                            temp = damping_factor * cavity[node][center_node] + (1 - damping_factor) * New_cavity_vec
                            temp /= tc.sum(temp)
                            difference = max(abs(temp - cavity[node][center_node]))
                            cavity[node][center_node] = temp
                            if is_inner_msg:
                                inner_msg[node] = temp
                        else:
                            difference = tc.tensor(0.0).to(device)
                    else:
                        default_msg = tc.zeros(8).to(device)
                        default_msg[0] = 1.0
                        default_msg[1] = 1.0
                        default_msg = default_msg / tc.sum(default_msg)
                        cavity[node][center_node] = default_msg
                        difference = tc.tensor(0.0).to(device)
                    if center_node == first_center_node and node == first_boundary_node:
                        difference_max = difference
                    elif difference > difference_max:
                        difference_max = difference
        if step >= 1:
            if difference_max.item() + 1e-6 >= tt:
                flag = False
        tt = difference_max.item()
        t2 = time.time()
        print("iteration step:", step + 1, ",   difference:", difference_max.item(), '    time:', t2 - t1)
        if difference_max <= epsilon:
            break
    print("skip:", skip, "  noskip:", noskip)
    print("skip2:", skip2, "  noskip2:", noskip2)
    if step == step_limit - 1:
        print('unconverged')
        converged = False
    return cavity, converged, egdelist2, step + 1