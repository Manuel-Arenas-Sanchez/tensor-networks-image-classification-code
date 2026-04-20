from __future__ import annotations

import math

import tensorkrowch as tk
import torch
import torch.nn as nn

from tn_dl.layers.base import BaseTensorNetworkLayer


class CustomTTN(tk.TensorNetwork):
    def __init__(
        self,
        sequence_length: int,
        local_dim: int,
        n_classes: int,
        bond_dim: int,
        out_position: int | None = None,
        boundary: str = "obc",
        name: str = "CustomTTN",
    ) -> None:
        super().__init__(name=name)

        self.sequence_length = sequence_length
        self.local_dim = local_dim
        self.n_classes = n_classes
        self.bond_dim = bond_dim
        self.out_position = out_position
        self.boundary = boundary

        self.input_nodes = []
        self.blocks_by_level = []
        self.padding_nodes_by_level = []
        self.root_node = None
        self.classifier_node = None

        self._build_tree()
        self._connect_levels()
    
    def _build_mps3_block(
        self,
        input_dim: int,
        output_dim: int,
        prefix: str,
    ):
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
            shape = (self.bond_dim ,input_dim),
            axes_names = ("right", "input"),
            name = f"{prefix}_left",
            network = self,
            init_method = "randn",
            std = init_std,
        )

        center_node = tk.ParamNode(
            shape = (self.bond_dim, self.bond_dim, input_dim, output_dim),
            axes_names = ("left", "right", "input", "output"),
            name = f"{prefix}_center",
            network = self,
            init_method = "randn",
            std = init_std,
        )
        right_node = tk.ParamNode(
            shape = (self.bond_dim, input_dim),
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
            "output_node": center_node,
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
    
    def _move_padding_nodes_to(self, x: torch.Tensor) -> None:
        """
        Los padding nodes son tk.Node con tensor fijo y no nn.Parameter. Por eso model.to(device)
        no los mueve automáticamente. Antes de contraer, los ponemos en el mismo device/dtype que la
        entrada.
        """
        for level_padding in self.padding_nodes_by_level:
            for pad_node in level_padding:
                if pad_node.tensor is None:
                    raise RuntimeError(
                        f"Padding node {pad_node.name} has no tensor materialized"
                    )
                if pad_node.tensor.device != x.device or pad_node.tensor.dtype != x.dtype:
                    pad_node.tensor = pad_node.tensor.to(
                    device = x.device,
                    dtype = x.dtype)

    def _build_tree(self) -> None:
        """
        Aqui se construye la jerarquía de bloques MPS con padding estructural.

        En cada nivel k:
            - num_blocks_k = ceil(num_sites_k / 3)
            - capacity_k = num_blocks_k * 3
            - padding_k = capacity_k - num_sites_k

        Se crean num_blocks_k bloques MPS de 3 nodos uniformes, y además padding_k nodos constantes
        (tk.Node fijos) que llenarían los edges sobrantes.

        Se guarda en paralelo:
            - self.blocks_by_level[k]         -> lista de diccionarios de bloque
            - self.padding_nodes_by_level[k]  -> lista de nodos de padding del nivel k
        """ 
        current_num_sites = self.sequence_length
        current_input_dim = self.local_dim
        level_idx = 0

        while current_num_sites > 1:
            num_blocks = math.ceil(current_num_sites / 3)
            capacity = num_blocks * 3
            padding_needed = capacity - current_num_sites

            # 1 Bloques reales
            level_blocks = []
            for block_idx in range(num_blocks):
                block = self._build_mps3_block(
                    input_dim = current_input_dim,
                    output_dim = self.bond_dim,
                    prefix = f"level{level_idx}_block{block_idx}",
                )
                level_blocks.append(block)
            self.blocks_by_level.append(level_blocks)

            # 2. Nodos de padding para los huecos
            level_padding = []
            for p_idx in range(padding_needed):
                pad = self._build_padding_node(
                    dim = current_input_dim,
                    name = f"level{level_idx}_pad{p_idx}",
                )
                level_padding.append(pad)
            self.padding_nodes_by_level.append(level_padding)

            # 3. Avanzar al siguiente nivel
            current_num_sites = num_blocks
            current_input_dim = self.bond_dim
            level_idx += 1

    def _connect_levels(self) -> None:
        """
        Aquí se conectan los edges input de cada nivel con lo que les corresponde:

        NIVEL 0:
            - Los primeros sequence_length edges son reales (se les atan data_nodes en set_data_nodes)
              Aqui no se hace nada con ellos
            - Los padding_needed_0 edges restantes se conectan a los nodos de padding del nivel 0.
        
        NIVEL K > 0:
            - Los primeros num_blocks_{k-1} edges se conectan a los output_edges de los bloques del
              nivel k-1.
            - Los padding_needed_k edges restantes se conectan a los nodos de padding del nivel k.
        """

        for level_idx, level_blocks in enumerate(self.blocks_by_level):
            level_padding = self.padding_nodes_by_level[level_idx]

            #Aplanar todos los input edges del nivel en orden
            input_edges = []
            for block in level_blocks:
                input_edges.extend(block["input_edges"])
            
            if level_idx == 0:
                # Los edges reales quedan libres (se enganchan en set_data_nodes).
                # Solo conectamos los de padding.
                for pad_idx, pad_node in enumerate(level_padding):
                    edge_idx = self.sequence_length + pad_idx
                    pad_node["output"] ^ input_edges[edge_idx]
            else:
                # Edges reales: conectar con outputs del nivel anterior
                prev_outputs = [b["output_edge"] for b in self.blocks_by_level[level_idx - 1]]

                for i, prev_out in enumerate(prev_outputs):
                    prev_out ^ input_edges[i]
                    
                # Edges de padding: conectar con los nodos de padding del nivel k
                for pad_idx, pad_node in enumerate(level_padding):
                    edge_idx = len(prev_outputs) + pad_idx
                    pad_node["output"] ^ input_edges[edge_idx]

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
        first_level = self.blocks_by_level[0]
        all_input_edges = []
        for block in first_level:
            all_input_edges.extend(block["input_edges"])

        real_edges = all_input_edges[: self.sequence_length]
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
        # NIVEL 0
        current_outputs = []

        for block in self.blocks_by_level[0]:
            current_outputs.append(self._contract_block(block))
        

        # NIVELES > 0
        for level_idx in range(1, len(self.blocks_by_level)):
            level_blocks = self.blocks_by_level[level_idx]
            level_padding = self.padding_nodes_by_level[level_idx]
            all_partners = list(current_outputs) + list(level_padding)

            next_outputs = []
            for block_idx, block in enumerate(level_blocks):
                block_partners = all_partners[3 * block_idx : 3 * block_idx + 3]
                next_outputs.append(self._contract_block(block, partners = block_partners))
            current_outputs = next_outputs
        
        assert len(current_outputs) == 1, (
            f"Se esperaba un único nodo raíz, se obtuvieron {len(current_outputs)}"
        )
        root_node = current_outputs[0]
        self.root_node = root_node
        return root_node
    
    def _contract_block(self, block, partners = None):
        """
        Contrae un bloque MPS-3 absorbiendo a sus 3 vecinos sobre el edge input.
        Devuelve el nodo resultante con un unico edge libre: el "output" del center.

        Si partners es None (nivel 0), se descubren vía neighbours("input").
        Si partners se pasa explicitamente (niveles > 0), se usa tal cual, porque
        neighbours() en esos niveles devuelve los nodos originales del nivel
        anterior, NO los composites.
        """
        left, center, right = block["nodes"]

        # Nota TK: node.neighbours(axis_name) devuelve el nodo conectado al edge con ese nombre de 
        # eje. Si la versión de TensorKrowch expone otra API para acceder al vecino 
        # (p. ej. left["input"].node2 o similar), habrá que ajustarlo aquí.
        if partners is None:
            partner_left = left.neighbours("input")
            partner_center = center.neighbours("input")
            partner_right = right.neighbours("input")
        else:
            partner_left, partner_center, partner_right = partners
        
        L = left @ partner_left
        C = center @ partner_center
        R = right @ partner_right

        return L @ C @ R

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

        self.network = CustomTTN(
            sequence_length = sequence_length,
            local_dim = local_dim,
            bond_dim = bond_dim,
            n_classes = n_classes,
            out_position = out_position,
            boundary = boundary,
        )
        self.classifier = nn.Linear(bond_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.network._move_padding_nodes_to(x)
        root_vec = self.network(x)  # deberia obtener un tensor [B, bond_dim]
        return self.classifier(root_vec)

