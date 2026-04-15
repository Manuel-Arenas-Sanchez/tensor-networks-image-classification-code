from __future__ import annotations

import tensorkrowch as tk

from tn_dl.layers.base import BaseTensorNetworkLayer


class MPSClassifier_1(BaseTensorNetworkLayer):
    # Creación de la tensor network con sus contracciones
    def __init__(self,
                 sequence_length: int,
                 local_dim: int,
                 n_classes: int,
                 bond_dim: int,
                 out_position: int | None = None,
                 boundary: str = "obc",
                 name: str = "MPSClassifier_1",) -> None:
        super().__init__()
        
        self.sequence_length = sequence_length
        self.local_dim = local_dim
        self.n_classes = n_classes
        self.bond_dim = bond_dim
        self.out_position = out_position
        self.boundary = boundary
        self.name = name

        self.net = tk.TensorNetwork(name=name)
        self.nodes = {}

        self.nodes[0] = tk.Node(
            shape=(local_dim, bond_dim),
            name="node_(0)",
            axes_names=("right", "input"),
            network=self.net,
        )

        self.nodes[sequence_length - 1] = tk.Node(
            shape=(bond_dim, local_dim),
            name=f"node_({sequence_length - 1})",
            axes_names=("left", "input"),
            network=self.net,
        )

        for i in range(1, sequence_length - 1):
            shapes = (
                (bond_dim, local_dim, bond_dim, local_dim)
                if i == sequence_length // 2
                else (bond_dim, local_dim, bond_dim)
            )
            axes_names = (
                ("left", "right", "up", "input")
                if i == sequence_length // 2
                else ("left", "right", "input")
            )
            self.nodes[i] = tk.Node(
                shape=shapes, name=f"node_({i})", axes_names=axes_names, network=self.net
            )

        for i in range(sequence_length - 1):
            self.nodes[i]["right"] ^ self.nodes[i + 1]["left"]

    def get_network(self):
        return self.net

    def get_node(self, i : int):
        return self.nodes[i]
    
    def forward(self, x):
        # [batch, sequence_length, local_dim]
        if x.ndim != 3:
            raise ValueError(
                "MPSClassifier_1 expects inputs shaped [batch, sequence_length, local_dim]."
            )
        if x.shape[1] != self.sequence_length:
            raise ValueError