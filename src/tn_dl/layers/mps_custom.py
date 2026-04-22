from __future__ import annotations

import math

import tensorkrowch as tk
import tensorkrowch.operations as op
import torch
import torch.nn as nn

from tn_dl.layers.base import BaseTensorNetworkLayer


class TTNLevel(tk.TensorNetwork):
    def __init__(
        self,
        num_input_sites: int,
        input_dim: int,
        output_dim: int,
        bond_dim: int,
        name: str,
    ) -> None:
        super().__init__(name=name)

        self.num_input_sites = num_input_sites
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim

        self.num_blocks = math.ceil(num_input_sites / 3)
        self.padding_needed = self.num_blocks * 3 - num_input_sites

        self.blocks = []
        self.padding_nodes = []


        self._build_level()
        self._connect_padding()
        
    def _build_mps3_block(self, prefix: str):
        """
        Contruyo un bloque base MPS con 3 nodos entrenables.

        Cada nodo es un tk.ParamNode: su tensor se registra como nn.Parameter del tk.TensorNetwork
        padre (self), de modo que:
            - queda materializado desde la construción (evitamos los None)
            - aparece en el model,parameters() automáticamente
            - no recibe gradientes en el backward

        Estructura interna:
            left: (bond, input)                         - axes (right, input)
            center: (bond, bond, input, output)         - axes (left, right, input, output)
            right: (bond, input)                        - axes (left, input)
        
        Bond internos:
            left["right"] ^ center["left"]
            center["right"] ^ right["left"]
        
        Edges libres del bloque:
          - 3 edges "input"  (uno por nodo) -> se enchufan a data_nodes / outputs
            del nivel anterior / nodos de padding en _connect_levels y
            set_data_nodes.
          - 1 edge "output"  en el center   -> sube al nivel siguiente, o es la
            raiz si no hay mas niveles.

        Inicializacion:
            init_method="randn", std=INIT_STD. Misma estrategia que la MPSLayer de
            referencia del repo (ver src/tn_dl/layers/mps.py). Un std pequeno
            (~0.02) mantiene las contracciones numericamente estables al subir
            niveles.
        """
        init_std = 0.02

        left_node = tk.ParamNode(
            shape = (self.bond_dim ,self.input_dim),
            axes_names = ("right", "input"),
            name = f"{prefix}_left",
            network = self,
            init_method = "randn",
            std = init_std,
        )

        center_node = tk.ParamNode(
            shape = (self.bond_dim, self.bond_dim, self.input_dim, self.output_dim),
            axes_names = ("left", "right", "input", "output"),
            name = f"{prefix}_center",
            network = self,
            init_method = "randn",
            std = init_std,
        )
        right_node = tk.ParamNode(
            shape = (self.bond_dim, self.input_dim),
            axes_names = ("left", "input"),
            name = f"{prefix}_right",
            network = self,
            init_method = "randn",
            std = init_std,
        )

        left_node["right"] ^ center_node["left"]
        center_node["right"] ^ right_node["left"]

        return {
            "nodes": [left_node, center_node, right_node],
            "input_edges": [
                left_node["input"],
                center_node["input"],
                right_node["input"],
            ],
            "output_edge": center_node["output"],
        }
    
    def _build_padding_node(self, dim: int, name: str) -> tk.Node:
        
        """
        Nodo constante, no entrenable, que se usa para rellenar posiciones sin dato real. 

        Se crea como tk.Node, por lo que su tensor no se registra como nn.Parameter y no recibe 
        gradientes. El tensor se inicializa a [1, 0, ..., 0], un vector neutro no-nulo para no
        colapsar el bloque a cero.
        """

        node = tk.Node(
            shape = (dim,),
            axes_names = ("output",),
            name = name,
            network = self,
        )
        neutral = torch.zeros(dim)
        neutral[0] = 1.0
        node.tensor = neutral
        return node

    def _build_level(self) -> None:
        for block_idx in range(self.num_blocks):
            self.blocks.append(
                self._build_mps3_block(prefix=f"{self.name}_block{block_idx}")
            )
        
        for pad_idx in range(self.padding_needed):
            self.padding_nodes.append(
                self._build_padding_node(
                    dim=self.input_dim,
                    name = f"{self.name}_pad{pad_idx}",
                )
            )
    
    def _connect_padding(self) -> None:
        input_edges = []
        for block in self.blocks:
            input_edges.extend(block["input_edges"])
        
        for pad_idx, pad_node in enumerate(self.padding_nodes):
            edge_idx = self.num_input_sites + pad_idx
            pad_node["output"] ^ input_edges[edge_idx]

    def _sync_padding_nodes_device(self) -> None:
        """
        Es un helper para moder los nodos padding a gpu, cuando quiero ejecutar el entrenamiento en
        gpu. Porque si no van directamente a cpu.
        """
        reference = next(self.parameters())
        for pad_node in self.padding_nodes:
            pad_node.tensor = pad_node.tensor.to(
                device = reference.device,
                dtype = reference.dtype
            )

    def set_data_nodes(self) -> None:
        """
        Esta función solo aplica al nivel 0. Le dice a TensorKrowch donde tiene que enchufar el
        batch real. El sitio es sobre los primeros sequence_length input edges del nivel 0.

        Diferencia de conceptos:
        `input_edges`: es el primer nivel, son puertos físicos reales
        `data_nodes`: los crea TensorKrowch cuando llamo a self.network(x)

        Los edges sobrantes del nivel 0 ya están conectados a nodos de padding estructural,
        (recomendable ver _connect_levels), así que no pasan aquí.
        """
        input_edges = []
        for block in self.blocks:
            input_edges.extend(block["input_edges"])

        real_edges = input_edges[: self.num_input_sites]
        super().set_data_nodes(real_edges, num_batch_edges = 1)
    
    def contract(self):
        """
        Aquí se ejecutan las contracciones del árbol completo, de leaves a raíz.

        Para cada bloque se contraen internamente los 3 nodos MPS (left, center, right) junto con
        sus "vecinos" sobre el edge input.

        Nivel 0:
            - Los vecinos son data_nodes (reales) o padding_nodes, ya enchufados
            por set_data_nodes / _connect_levels.
            - _contract_block los descubre vía neighbours("input").

        Niveles > 0:
            - Los vecinos son los composites del nivel anterior + los padding_nodes
              del nivel actual.
            - Hay que pasarlos EXPLICITAMENTE a _contract_block, porque
              left.neighbours("input") seguiría devolviendo el center original del
              nivel anterior (TK no reruta .node1/.node2 al crear el composite);
              pero el @ sobre el composite sí detecta el edge compartido porque el
              composite heredó la misma referencia de edge (ver TK
              operations.py:3961 — new_edges.append(nodes[i][idx])).
        """
        self._sync_padding_nodes_device()

        outputs = []

        for block in self.blocks:
            outputs.append(self._contract_block(block))
        
        return op.stack(outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y.movedim(0, 1)

    def _contract_block(self, block):
        """
        Contrae un bloque MPS-3 absorbiendo a sus 3 vecinos sobre el edge input.
        Devuelve el nodo resultante con un unico edge libre: el "output" del center.

        Si partners es None (nivel 0), se descubren vía neighbours("input").
        Si partners se pasa explicitamente (niveles > 0), se usa tal cual, porque
        neighbours() en esos niveles devuelve los nodos originales del nivel
        anterior, NO los composites.
        """
        left, center, right = block["nodes"]

        partner_left = left.neighbours("input")
        partner_center = center.neighbours("input")
        partner_right = right.neighbours("input")
        
        L = left @ partner_left
        C = center @ partner_center
        R = right @ partner_right

        return L @ C @ R


class HybridCustomTTN(nn.Module):
  
    def __init__(
            self,
            sequence_length: int,
            local_dim: int,
            bond_dim: int,
        ) -> None:
        super().__init__()

        self.sequence_length = sequence_length
        self.local_dim = local_dim
        self.bond_dim = bond_dim

        levels = []
        transforms = []

        current_sites = sequence_length
        current_dim = local_dim
        level_idx = 0

        while current_sites > 1:
            level = TTNLevel(
                num_input_sites = current_sites,
                input_dim = current_dim,
                output_dim = bond_dim,
                bond_dim = bond_dim,
                name = f"Hybrid_TTN_Level{level_idx}",
            )
            levels.append(level)

            next_sites = math.ceil(current_sites / 3)

            if next_sites > 1:
                transforms.append(
                    nn.Sequential(
                        nn.LayerNorm(bond_dim),
                        nn.Linear(bond_dim, bond_dim),
                        nn.ReLU(),
                    )
                )
            current_sites = next_sites
            current_dim = bond_dim
            level_idx += 1
        
        self.levels = nn.ModuleList(levels)
        self.transforms = nn.ModuleList(transforms)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for level_idx, level in enumerate(self.levels):
            x = level(x)

            if level_idx < len(self.transforms):
                x = self.transforms[level_idx](x)
        
        return x.squeeze(1)

class MPSClassifier_1(BaseTensorNetworkLayer):
    def __init__(
        self,
        sequence_length: int,
        local_dim: int,
        n_classes: int,
        bond_dim: int,
        out_position: int | None = None,
        boundary: str = "obc",
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.local_dim = local_dim
        self.n_classes = n_classes
        self.bond_dim = bond_dim
        self.out_position = out_position
        self.boundary = boundary

        self.network = HybridCustomTTN(
            sequence_length = sequence_length,
            local_dim = local_dim,
            bond_dim = bond_dim,
        )
        self.classifier = nn.Linear(bond_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        root_vec = self.network(x) # deberia obtener un tensor [B, bond_dim]
        return self.classifier(root_vec)
