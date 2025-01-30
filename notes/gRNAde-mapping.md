### Visualization 

![[gRNAde-markmap.png]]

### **1. Input Features (featurizer.py)**
**Code Reference**: `RNAGraphFeaturizer.__call__()` (lines 63-150)

node_s = internal_coords_feat  # [N, C, 15]
Scalar features - 3 atoms × 5 features (cos/sin angles + log lengths) 
	- [cos(δ), sin(δ), cos(θ), sin(θ), log(l)] for each backbone atom

node_v = internal_vecs_feat    # [N, C, 4, 3] - 4 vectors per node:
Vector features - 4 normalized displacement vectors: 
	- 1. P→C4', 2. C4'→N, 3. Forward (C4'_{i}→C4'_{i+1}), 4. Backward (C4'_{i}→C4'_{i-1})

edge_s = torch.cat([edge_rbf, edge_posenc, torch.log(edge_lengths)], dim=-1)   # [E, C, 131] 
Edge scalars - 32 RBF + 32 posenc + 3 log distances + 64 (edge_h_dim)

edge_v: # [E, C, 3, 3] - displacement vectors between backbone atoms (P, C4', N)
Edge vectors - Normalized displacement vectors for 3 backbone atoms

Note here:  
- $N$: Number of nodes (nucleotides). 
- $C$: Number of conformations.
- $E$: Number of edges.

### **2. Encoder Phase (models.py)**
**Code Reference**: `AutoregressiveMultiGNNv1.forward()` (lines 64-74)

***Initial embedding (models.py lines 26-35)***
self.W_v = GVP((15,4) → (128,16))  # node_in_dim → node_h_dim
self.W_e = GVP((131,3) → (64,4))   # edge_in_dim → edge_h_dim

h_V = (N, C, 15) → (N, C, 128)   # scalar
        (N, C, 4,3) → (N, C, 16,3) # vector

h_E = (E, C, 131) → (E, C, 64)    # scalar
        (E, C, 3,3) → (E, C, 4,3)  # vector

***Message passing (layers.py lines 238-278)***
for layer in self.encoder_layers:  # 4x MultiGVPConvLayer
    h_V = layer(h_V, edge_index, h_E)  # h_V_new = σ( W1 * (h_V, ⨁_{j∈N(i)} M(h_V_j, h_E_ij)) )

_Key mechanism_: Each `MultiGVPConvLayer` preserves shape while mixing information:

*layers.py lines 244-248*
message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))  # [E, C, 128+64+128=320]
message = self.message_func(message)  # GVP preserves dimensionality

### **3. Pooling (models.py)**
**Code Reference**: `AutoregressiveMultiGNNv1.pool_multi_conf()` (lines 155-177)

**Mask invalid conformers (e.g., padding)**
mask = mask_confs.unsqueeze(2)  # [N, C, 1]

**Scalar pooling**
h_V0 = h_V[0] * mask            # [N, C, 128]
h_V0 = h_V0.sum(dim=1) / n_conf_true  # [N, 128]

**Vector pooling**
mask = mask.unsqueeze(3)        # [N, C, 1, 1] 
h_V1 = h_V[1] * mask            # [N, C, 16, 3]
h_V1 = h_V1.sum(dim=1) / n_conf_true.unsqueeze(2)  # [N, 16, 3]


### 4. **Post-Pooling Workflow**

#### **4.1. Input to Decoder**

**Features** (from `AutoregressiveMultiGNNv1.forward()`):

- **Node Features**:
    - `node_s_pooled`: `(N, 128)`
    - `node_v_pooled`: `(N, 16, 3)`
        
- **Edge Features**:
    - `edge_s_pooled`: `(E, 64)`
    - `edge_v_pooled`: `(E, 4, 3)`
        
- **Autoregressive Context**:
    - `seq`: Target sequence `(N,)` (int tensor) → Embedded to `h_S = (N, 4)`

#### **4.2. Decoder Layers** (`models.py`, `layers.py`)

**Key Code**:
**models.py (lines 79-87)**
self.decoder_layers = nn.ModuleList([
    GVPConvLayer(node_h_dim, edge_h_dim, autoregressive=True)
    for _ in range(num_layers)
])

**During forward pass:**
h_S = self.W_s(seq)  # Embed sequence: (N,) → (N, 4)
h_S = h_S[edge_index[0]]  # Propagate to edges: (E, 4)
h_S[edge_index[0] >= edge_index[1]] = 0  # Mask future nodes
h_E = (torch.cat([edge_s, h_S], dim=-1), edge_v)  # (E, 64+4=68)


**Steps**:
1. **Edge Context Injection**:
    - Augment edge scalars with sequence embeddings:  
        `edge_s = (E, 64) → (E, 68)`
    - _Purpose_: Condition edges on autoregressive sequence history.
        
2. **Autoregressive Message Passing**:
    - For each of 4 decoder layers (`GVPConvLayer`):
        - **Message Function**: Mixes node/edge features while enforcing causality (src < dst).
			**layers.py (lines 89-106)**
			mask = src < dst  # Only allow messages from past→future
			message = tuple_cat(s_j, v_j, edge_attr, s_i, v_i)

#### **4.3. Output Layer** (`models.py`)

**Final Transformation**:
**models.py (line 45)**
self.W_out = GVP((128,16), (4,0))  # Scalar-only output
logits = self.W_out(h_V)  # (N, 4)

**Output**:
- `logits`: `(N, 4)` → Log probabilities for {A, G, C, U} at each position.
    
---

### **4.4. Sampling** (`models.py`)

**Autoregressive Decoding** (`sample()` method):
for i in range(num_nodes):
    # Mask edges to only update node i
    edge_mask = edge_index[1] == i  # (E,)
    logits_i = model(h_V, edge_index[:, edge_mask], h_E[edge_mask])
    seq[i] = Categorical(logits_i).sample()

**Key Constraints**:

- **Causal Masking**: Nodes can only attend to predecessors (`src < i`).
- **Teacher Forcing**: Uses ground truth `seq[0:i]` during training.
    
---
### **Hierarchical Summary**
Post-Pooling Phase
├── Decoder Input  
│   ├── Node Features: (N,128), (N,16,3)  
│   ├── Edge Features: (E,64 → 68), (E,4,3)  
│   └── Sequence Context: (N,4)  
├── Autoregressive Layers (x4)  
│   ├── Edge Masking: src < dst  
│   ├── Message Aggregation: Geometric + Sequence  
│   └── Node Updates: LayerNorm(Residual + Messages)  
├── Output Projection  
│   └── GVP: (128,16) → (4,0)  
└── Sampling  
    ├── Stepwise Generation: i = 0 → N-1  
    └── Causal Constraint: No future leakage  


### Markmap Code: 

# gRNAde Forward Pass

## Encoder Phase

### Input Features (`featurizer.py`)
- **Nodes**:
  - `node_s`: (N, C, 15)  
    *3 atoms × [cos(δ), sin(δ), cos(θ), sin(θ), log(l)]*
  - `node_v`: (N, C, 4, 3)  
    *[P→C4', C4'→N, forward, backward] vectors*
- **Edges**:
  - `edge_s`: (E, C, 131)  
    *32 RBF + 32 posenc + 3 log-dist*
  - `edge_v`: (E, C, 3, 3)  
    *P/C4'/N displacement vectors*
  - `edge_index`: (2, E) (KNN graph)

### Embedding (`models.py`)
- **Node GVP**: (15,4) → (128,16)
  - `node_s → (N,C,128)`
  - `node_v → (N,C,16,3)`
- **Edge GVP**: (131,3) → (64,4)
  - `edge_s → (E,C,64)`
  - `edge_v → (E,C,4,3)`

### Message Passing (`layers.py`)
- 4x MultiGVPConvLayer
  - Preserves shapes:
    - Nodes: (N,C,128), (N,C,16,3)
    - Edges: (E,C,64), (E,C,4,3)
  - Mechanism:  
    ```python
    message = tuple_cat(s_j, v_j, edge_attr, s_i, v_i)
    ```

### Pooling (`models.py`)
- **Masked Mean**:
  - Nodes: (N,128), (N,16,3)
  - Edges: (E,64), (E,4,3)

## Decoder Phase

### Input to Decoder
- **Pooled Features**:
  - Nodes: (N,128), (N,16,3)
  - Edges: (E,64), (E,4,3)
- **Sequence Context**:
  - `h_S = Embed(seq) → (N,4)`

### Autoregressive Layers (`models.py`)
- 4x GVPConvLayer
- **Key Operations**:
  1. Edge Context Injection:
     - `edge_s = (E,64+4=68)`
  2. Causal Masking:
     ```python
     mask = src < dst  # layers.py:94-97
     ```
  3. Message Aggregation:
     - Geometric + Sequence features

### Output Projection
- **Final GVP**: (128,16) → (4,0)
  - `logits: (N,4)`

### Sampling (`models.py`)
- **Autoregressive Decoding**:
  ```python
  for i in 0..N-1:
      edge_mask = edge_index[1] == i
      seq[i] ∼ Categorical(logits_i)  # lines 113-148