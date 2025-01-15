import logging, numpy as np


class Graph:
    # max_hop几阶邻域
    def __init__(self, dataset, n_node, max_hop=2, dilation=1):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation
        self.n_node = n_node

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'coco':
            num_node = self.n_node
            self_link = [(i, i) for i in range(num_node)]

            if num_node == 32:
                neighbor_link = [(0, 1), (0, 18), (0, 22), (18, 19), (19, 20), (20, 21), (22, 23), (23, 24), (24, 25),
                                 (1, 2), (2, 3), (2, 4), (2, 11), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (7, 10),
                                 (11, 12),
                                 (12, 13), (13, 14), (14, 15), (15, 16), (14, 17), (3, 26), (26, 27), (26, 28),
                                 (26, 29),
                                 (26, 30), (26, 31)
                                 ]
            else:
                neighbor_link = [(0, 1), (0, 2), (0, 3), (0, 4)]

            self.edge = self_link + neighbor_link
            self.center = 0
            if num_node == 32:
                connect_joint = np.array(
                    [3, 0, 1, 2, 2, 4, 5, 6, 7, 8, 7, 2, 11, 12, 13, 14, 15, 14, 0, 18, 19, 20, 0, 22, 23, 24, 3, 26, 26,
                     26, 26, 26])
            else:
                connect_joint = np.array(
                    [1, 0, 0, 0, 0])

            parts = [
                # np.array([4, 5, 6, 7, 8, 9, 10]),  # left_arm
                # np.array([11, 12, 13, 14, 15, 16, 17]),  # right_arm
                # np.array([18, 19, 20, 21]),  # left_leg
                # np.array([22, 23, 24, 25]),  # right_leg
                # np.array([0, 1, 2, 3, 26, 27, 28, 29, 30, 31]),  # torso + head
            ]
        else:
            num_node, neighbor_link, connect_joint, parts = 0, [], [], []
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:  # 构造邻接矩阵是对称矩阵无向图
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]  # 计算矩阵次方，矩阵A的[0，3]次方,构造M阶邻域
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d  # 形成一个array(32,32)的表示max_hop阶邻居距离的矩阵
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:  # 将在valid_hop距离之内的全都置为1
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):  # A[0]只表示自身邻域，A[1]只表示一阶邻域，A[2]只表示二阶邻域的特征
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD
