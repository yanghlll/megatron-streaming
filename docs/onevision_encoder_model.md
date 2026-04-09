# OneVisionEncoderModel — Data Flow

```mermaid
flowchart TD
    %% ============================================================
    %% Style Definitions
    %% ============================================================
    classDef main_node fill:#ffffff,stroke:#333,stroke-width:2px;
    classDef sub_module fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef norm_node fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef linear_node fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef attn_node fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef mlp_node fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef bda_node fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef io_node fill:#eeeeee,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5;
    classDef weight_node fill:#ffe0b2,stroke:#ef6c00,stroke-width:1px,font-size:10px;
    classDef act_node fill:#fff3e0,stroke:#ef6c00,stroke-width:1px,stroke-dasharray: 2 2;
    classDef flag_node fill:#e8eaf6,stroke:#3949ab,stroke-width:2px,font-style:italic;

    %% ============================================================
    %% Inputs
    %% ============================================================
    InputX(("Input: x<br/>Shape: [total_patches, C x P x P]")):::io_node
    InputGridTHW(("Input: grid_thw<br/>Shape: [batch, 3]")):::io_node
    InputPatchPos(("Input: patch_positions<br/>Shape: [total_patches, 3]<br/>(optional)")):::io_node

    %% ========================================
    %% Step 1: Frame Windowing
    %% ========================================
    subgraph FrameWindow ["Frame Windowing"]
        direction TB
        FW_Check{"t > frame_windows_size<br/>(default: 4)?"}:::main_node
        FW_Split["Split temporal dim into<br/>windows of frame_windows_size<br/>+ optional remainder"]:::main_node
        FW_Pass["Keep grid_thw as-is"]:::main_node
        GridTHW_Out(("grid_thw<br/>Shape: [batch_expanded, 3]")):::io_node

        FW_Check -- Yes --> FW_Split --> GridTHW_Out
        FW_Check -- No --> FW_Pass --> GridTHW_Out
    end
    InputGridTHW --> FW_Check

    %% ========================================
    %% Step 2: _SCATTER_BEFORE_PATCH_EMBED branch
    %% ========================================
    Flag{"_SCATTER_BEFORE_PATCH_EMBED<br/>(env: SCATTER_BEFORE_PATCH_EMBED)"}:::flag_node
    InputX --> Flag

    %% ── PATH A: SCATTER=1 (optimised) ──────────────────────────
    subgraph SP_Scatter_X ["_scatter_x  [SCATTER=1 only]<br/>Scatter BEFORE patch_embed"]
        direction TB
        SPX_Check{"config.sequence_parallel?"}:::main_node
        SPX_Pad["Pad x to next multiple<br/>of tp_size<br/>(sp_pad_size computed here)"]:::main_node
        SPX_ScatterOp["scatter_to_sequence_parallel_region<br/>(split x along token dim)"]:::main_node
        SPX_Skip["Pass through<br/>(no scatter)"]:::main_node

        SPX_Check -- Yes --> SPX_Pad --> SPX_ScatterOp
        SPX_Check -- No --> SPX_Skip
    end
    Flag -- "=1" --> SPX_Check

    SP_X_Scattered(("x  [SCATTER=1]<br/>Shape: [s/tp, C x P x P]<br/>(or [s, C x P x P] if no SP)")):::io_node
    SPX_ScatterOp --> SP_X_Scattered
    SPX_Skip --> SP_X_Scattered

    %% ── PATH B: SCATTER=0 (original) ───────────────────────────
    X_Full(("x  [SCATTER=0]<br/>Shape: [s, C x P x P]")):::io_node
    Flag -- "=0" --> X_Full

    %% ========================================
    %% Step 3: Patch Embedding
    %% ========================================
    subgraph PE_TP ["ParallelPatchEmbed (TP)<br/>[incompatible with SCATTER=1]"]
        direction LR
        PE_TP_In(("Input")):::io_node
        PE_TP_Proj["ColumnParallelLinear<br/>(gather_output=True)"]:::linear_node
        PE_TP_W["Weight<br/>Shape: [h/tp, C x P x P]"]:::weight_node
        PE_TP_Out(("Output<br/>Shape: [*, h]")):::io_node

        PE_TP_In --> PE_TP_Proj
        PE_TP_W -.-> PE_TP_Proj
        PE_TP_Proj --> PE_TP_Out
    end

    subgraph PE_Linear ["TorchLinearPatchEmbed"]
        direction LR
        PE_L_In(("Input")):::io_node
        PE_L_Proj["nn.Linear<br/>(bias=False)"]:::linear_node
        PE_L_W["Weight<br/>Shape: [h, C x P x P]"]:::weight_node
        PE_L_Out(("Output<br/>Shape: [*, h]")):::io_node

        PE_L_In --> PE_L_Proj
        PE_L_W -.-> PE_L_Proj
        PE_L_Proj --> PE_L_Out
    end

    subgraph PE_Conv ["PatchEmbed (Conv2d)"]
        direction TB
        PE_C_Reshape["Reshape<br/>[*, C x P x P] → [*, C, P, P]"]:::main_node
        PE_C_Conv["Conv2d<br/>(kernel=P, stride=P, bias=False)"]:::linear_node
        PE_C_W["Weight<br/>Shape: [h, C, P, P]"]:::weight_node
        PE_C_Flat["Reshape<br/>[*, h, 1, 1] → [*, h]"]:::main_node
        PE_C_Out(("Output<br/>Shape: [*, h]")):::io_node

        PE_C_Reshape --> PE_C_Conv
        PE_C_W -.-> PE_C_Conv
        PE_C_Conv --> PE_C_Flat --> PE_C_Out
    end

    SP_X_Scattered --> PE_TP_In
    SP_X_Scattered --> PE_L_In
    SP_X_Scattered --> PE_C_Reshape
    X_Full --> PE_TP_In
    X_Full --> PE_L_In
    X_Full --> PE_C_Reshape

    PE_Emb(("x after patch_embed<br/>Shape: [s/tp, h]  (SCATTER=1)<br/>Shape: [s, h]   (SCATTER=0)")):::io_node
    PE_TP_Out --> PE_Emb
    PE_L_Out --> PE_Emb
    PE_C_Out --> PE_Emb

    %% ========================================
    %% Step 4: 3D Rotary Position Embedding (REPLICATED on every rank)
    %% ========================================
    subgraph RoPE_Module ["VideoRotaryEmbeddingSplit466 (3D RoPE)<br/>Always computed on FULL sequence"]
        direction TB

        RoPE_Select{"patch_positions<br/>provided?"}:::main_node

        %% Path A: from patch_positions
        RoPE_P_Tpos["t_pos = positions[:, 0]"]:::attn_node
        RoPE_P_Hpos["h_pos = positions[:, 1]"]:::attn_node
        RoPE_P_Wpos["w_pos = positions[:, 2]"]:::attn_node
        RoPE_P_FT["ft = outer(t_pos, inv_freq_t)<br/>4/16 of half"]:::attn_node
        RoPE_P_FH["fh = outer(h_pos, inv_freq_h)<br/>6/16 of half"]:::attn_node
        RoPE_P_FW["fw = outer(w_pos, inv_freq_w)<br/>6/16 of half"]:::attn_node
        RoPE_P_Cat["cat(ft, fh, fw)"]:::attn_node

        %% Path B: from grid_thw
        RoPE_G_Freq["Compute freq tables<br/>ft, fh, fw per (t, h, w)"]:::attn_node
        RoPE_G_Idx["Build position indices<br/>t_ids, h_ids, w_ids"]:::attn_node
        RoPE_G_Cat["cat(ft[t_ids], fh[h_ids], fw[w_ids])"]:::attn_node
        RoPE_G_Block["convert_rope_to_block_layout<br/>(row-major to 2x2 block order)"]:::attn_node

        %% Frequency buffer weights
        RoPE_InvT["inv_freq_t<br/>Shape: [4 x unit]"]:::weight_node
        RoPE_InvH["inv_freq_h<br/>Shape: [6 x unit]"]:::weight_node
        RoPE_InvW["inv_freq_w<br/>Shape: [6 x unit]"]:::weight_node

        %% Path A connections
        RoPE_Select -- Yes --> RoPE_P_Tpos
        RoPE_P_Tpos --> RoPE_P_FT
        RoPE_P_Hpos --> RoPE_P_FH
        RoPE_P_Wpos --> RoPE_P_FW
        RoPE_P_FT --> RoPE_P_Cat
        RoPE_P_FH --> RoPE_P_Cat
        RoPE_P_FW --> RoPE_P_Cat

        %% Path B connections
        RoPE_Select -- No --> RoPE_G_Freq
        RoPE_G_Freq --> RoPE_G_Idx --> RoPE_G_Cat --> RoPE_G_Block

        %% Weight connections
        RoPE_InvT -.-> RoPE_P_FT
        RoPE_InvH -.-> RoPE_P_FH
        RoPE_InvW -.-> RoPE_P_FW
        RoPE_InvT -.-> RoPE_G_Freq
        RoPE_InvH -.-> RoPE_G_Freq
        RoPE_InvW -.-> RoPE_G_Freq

        %% Outputs
        RoPE_Out(("rotary_pos_emb<br/>Shape: [s, half]<br/>(replicated, full sequence)")):::io_node
        RoPE_P_Cat --> RoPE_Out
        RoPE_G_Block --> RoPE_Out
    end
    GridTHW_Out --> RoPE_Select
    InputPatchPos -.-> RoPE_Select

    %% ========================================
    %% Step 5: Build cu_seqlens (REPLICATED on every rank)
    %% ========================================
    subgraph CuSeqlens ["Build cu_seqlens<br/>Always computed on FULL sequence"]
        direction TB
        CS_Compute["Cumulative sum of<br/>tokens_per_sample<br/>(t x h x w per sample)"]:::main_node
        CS_Tensor["torch.tensor<br/>(dtype=int32)"]:::main_node
        CuSeq_Out(("cu_seqlens<br/>Shape: [batch + 1]<br/>(replicated, full sequence)")):::io_node

        CS_Compute --> CS_Tensor --> CuSeq_Out
    end
    GridTHW_Out --> CS_Compute

    %% ========================================
    %% Step 6: Post-embed SP preparation  (diverges by path)
    %% ========================================
    Flag2{"_SCATTER_BEFORE_PATCH_EMBED"}:::flag_node
    PE_Emb --> Flag2

    %% ── PATH A continued: SCATTER=1 ─────────────────────────────
    subgraph SP_Pad_RoPE ["_pad_rope_and_cu_seqlens  [SCATTER=1 only]<br/>Pad RoPE + cu_seqlens (x already scattered)"]
        direction TB
        SPP_Check{"config.sequence_parallel &&<br/>sp_pad_size > 0?"}:::main_node
        SPP_PadRoPE["Pad rotary_pos_emb<br/>along token dim (sp_pad_size)"]:::main_node
        SPP_PadCuSeq["Append dummy entry<br/>to cu_seqlens (+sp_pad_size)"]:::main_node
        SPP_Skip["Pass through<br/>(no padding needed)"]:::main_node

        SPP_Check -- Yes --> SPP_PadRoPE --> SPP_PadCuSeq
        SPP_Check -- No --> SPP_Skip
    end
    Flag2 -- "=1" --> SPP_Check
    RoPE_Out --> SPP_Check
    CuSeq_Out --> SPP_Check

    Unsqueeze_A["Unsqueeze dim=1<br/>[s/tp, h] → [s/tp, 1, h]"]:::main_node
    Flag2 -- "=1" --> Unsqueeze_A

    %% ── PATH B continued: SCATTER=0 ─────────────────────────────
    subgraph SP_Scatter_Full ["_scatter_for_sequence_parallel  [SCATTER=0 only]<br/>Pad + scatter x AND pad RoPE + cu_seqlens"]
        direction TB
        SPS_Check{"config.sequence_parallel?"}:::main_node
        SPS_PadX["Pad x along token dim<br/>(sp_pad_size computed here)"]:::main_node
        SPS_PadRoPE["Pad rotary_pos_emb<br/>along token dim"]:::main_node
        SPS_PadCuSeq["Append dummy entry<br/>to cu_seqlens"]:::main_node
        SPS_ScatterX["scatter_to_sequence_parallel_region(x)<br/>[s_padded, 1, h] → [s_padded/tp, 1, h]"]:::main_node
        SPS_Skip["Pass through<br/>(no scatter)"]:::main_node

        SPS_Check -- Yes --> SPS_PadX
        SPS_PadX --> SPS_PadRoPE --> SPS_PadCuSeq --> SPS_ScatterX
        SPS_Check -- No --> SPS_Skip
    end
    Flag2 -- "=0" --> SPS_Check
    RoPE_Out --> SPS_Check
    CuSeq_Out --> SPS_Check

    Unsqueeze_B["Unsqueeze dim=1<br/>[s, h] → [s, 1, h]"]:::main_node
    Flag2 -- "=0" --> Unsqueeze_B
    Unsqueeze_B --> SPS_Check

    %% Merge paths into common state for transformer
    X_Local(("hidden_states  [both paths]<br/>Shape: [s_local, 1, h]<br/>s_local = s/tp (SP) or s (no SP)")):::io_node
    Unsqueeze_A --> X_Local
    SPS_ScatterX --> X_Local
    SPS_Skip --> X_Local

    RoPE_Ready(("rotary_pos_emb<br/>Shape: [s_padded, half]<br/>(replicated)")):::io_node
    SPP_PadCuSeq --> RoPE_Ready
    SPP_Skip --> RoPE_Ready
    SPS_PadCuSeq --> RoPE_Ready

    CuSeq_Ready(("cu_seqlens<br/>(padded, replicated)")):::io_node
    SPP_PadCuSeq --> CuSeq_Ready
    SPP_Skip --> CuSeq_Ready
    SPS_PadCuSeq --> CuSeq_Ready
    SPS_Skip --> CuSeq_Ready

    %% ========================================
    %% Step 7: Pre-LayerNorm
    %% ========================================
    subgraph PreLN ["pre_layernorm (TENorm)"]
        direction LR
        PreLN_Op["LayerNorm / RMSNorm<br/>(eps=1e-4)"]:::norm_node
        PreLN_Gamma["Weight (gamma)<br/>Shape: [h]"]:::weight_node
        PreLN_Beta["Bias (beta)<br/>Shape: [h]"]:::weight_node
        PreLN_Op -.-> PreLN_Gamma
        PreLN_Op -.-> PreLN_Beta
        PreLN_Out(("hidden_states<br/>Shape: [s_local, 1, h]")):::io_node
        PreLN_Op --> PreLN_Out
    end
    X_Local --> PreLN_Op

    %% ========================================
    %% Step 8: TransformerBlock - Self-Attention
    %% ========================================
    subgraph Block1 ["Block 1: Self-Attention (TransformerLayer)"]
        direction TB

        TF_Residual1(("Residual 1<br/>Shape: [s_local, 1, h]")):::io_node

        %% QKV Linear (fused LN + ColParallel)
        subgraph QKV_Fused ["TELayerNormColumnParallelLinear (QKV)"]
            direction LR
            QKV_In(("Input<br/>Shape: [s_local, 1, h]")):::io_node
            QKV_LN["Fused LayerNorm"]:::norm_node
            QKV_MatMul["MatMul<br/>(ColumnParallel)"]:::linear_node
            QKV_W["Weight<br/>Shape: [3h/tp, h]"]:::weight_node
            QKV_B["Bias<br/>Shape: [3h/tp]"]:::weight_node
            QKV_Out(("Output<br/>Shape: [s_local, 1, 3h/tp]")):::io_node

            QKV_In --> QKV_LN --> QKV_MatMul
            QKV_W -.-> QKV_MatMul
            QKV_MatMul --> QKV_Out
            QKV_B -.-> QKV_Out
        end

        %% Core Attention
        subgraph CoreAttn ["TEDotProductAttention"]
            direction TB
            CA_Split["Split Q, K, V"]:::attn_node
            CA_RoPE["Apply 3D RoPE<br/>(4:6:6 T:H:W split)"]:::attn_node
            CA_Flash["FlashAttention Kernel<br/>(qkv_format=thd,<br/>packed_seq via cu_seqlens)"]:::attn_node

            CA_Split --> CA_RoPE --> CA_Flash
        end

        %% Output projection (RowParallel)
        subgraph O_Proj ["TERowParallelLinear (Proj)"]
            direction LR
            O_In(("Input<br/>Shape: [s_local, 1, h/tp]")):::io_node
            O_MatMul["MatMul<br/>(RowParallel)"]:::linear_node
            O_W["Weight<br/>Shape: [h, h/tp]"]:::weight_node
            O_Out(("Output<br/>Shape: [s_local, 1, h]")):::io_node

            O_In --> O_MatMul
            O_W -.-> O_MatMul
            O_MatMul --> O_Out
        end

        %% Bias-Dropout-Add
        subgraph BDA1 ["self_attn_bda"]
            direction TB
            BDA1_In(("Attn Output<br/>Shape: [s_local, 1, h]")):::io_node
            BDA1_Res(("Residual<br/>Shape: [s_local, 1, h]")):::io_node
            BDA1_Add["Residual + Attn Output"]:::bda_node
            BDA1_Drop["Dropout"]:::bda_node

            BDA1_In --> BDA1_Add
            BDA1_Res --> BDA1_Add
            BDA1_Add --> BDA1_Drop
        end

        %% Connections
        TF_Residual1 --> QKV_In
        QKV_Out --> CA_Split
        CA_Flash --> O_In
        O_Out --> BDA1_In
        TF_Residual1 -.-> BDA1_Res
        BDA1_Drop --> Hidden1(("Hidden States 1<br/>Shape: [s_local, 1, h]")):::io_node
    end
    PreLN_Out --> TF_Residual1
    RoPE_Ready -.-> CA_RoPE
    CuSeq_Ready -.-> CA_Flash

    %% ========================================
    %% Step 9: TransformerBlock - MLP
    %% ========================================
    subgraph Block2 ["Block 2: MLP (TransformerLayer)"]
        direction TB

        TF_Residual2(("Residual 2<br/>Shape: [s_local, 1, h]")):::io_node

        %% FC1 (fused LN + ColParallel)
        subgraph FC1_Fused ["TELayerNormColumnParallelLinear (FC1)"]
            direction LR
            FC1_In(("Input<br/>Shape: [s_local, 1, h]")):::io_node
            FC1_LN["Fused LayerNorm"]:::norm_node
            FC1_MatMul["MatMul<br/>(ColumnParallel)"]:::linear_node
            FC1_W["Weight<br/>Shape: [4h/tp, h]"]:::weight_node
            FC1_B["Bias<br/>Shape: [4h/tp]"]:::weight_node
            FC1_Out(("Output<br/>Shape: [s_local, 1, 4h/tp]")):::io_node

            FC1_In --> FC1_LN --> FC1_MatMul
            FC1_W -.-> FC1_MatMul
            FC1_MatMul --> FC1_Out
            FC1_B -.-> FC1_Out
        end

        %% Activation
        subgraph ActFunc ["Activation"]
            direction LR
            Act_In(("Input<br/>Shape: [s_local, 1, 4h/tp]")):::io_node
            Act_GeLU["GeLU"]:::act_node
            Act_Out(("Output<br/>Shape: [s_local, 1, 4h/tp]")):::io_node
            Act_In --> Act_GeLU --> Act_Out
        end

        %% FC2 (RowParallel)
        subgraph FC2_Proj ["TERowParallelLinear (FC2)"]
            direction LR
            FC2_In(("Input<br/>Shape: [s_local, 1, 4h/tp]")):::io_node
            FC2_MatMul["MatMul<br/>(RowParallel)"]:::linear_node
            FC2_W["Weight<br/>Shape: [h, 4h/tp]"]:::weight_node
            FC2_Out(("Output<br/>Shape: [s_local, 1, h]")):::io_node

            FC2_In --> FC2_MatMul
            FC2_W -.-> FC2_MatMul
            FC2_MatMul --> FC2_Out
        end

        %% Bias-Dropout-Add
        subgraph BDA2 ["mlp_bda"]
            direction TB
            BDA2_In(("MLP Output<br/>Shape: [s_local, 1, h]")):::io_node
            BDA2_Res(("Residual<br/>Shape: [s_local, 1, h]")):::io_node
            BDA2_Add["Residual + MLP Output"]:::bda_node
            BDA2_Drop["Dropout"]:::bda_node

            BDA2_In --> BDA2_Add
            BDA2_Res --> BDA2_Add
            BDA2_Add --> BDA2_Drop
        end

        %% Connections
        Hidden1 --> TF_Residual2
        TF_Residual2 --> FC1_In
        FC1_Out --> Act_In
        Act_Out --> FC2_In
        FC2_Out --> BDA2_In
        TF_Residual2 -.-> BDA2_Res
        BDA2_Drop --> Hidden2(("Hidden States 2<br/>Shape: [s_local, 1, h]")):::io_node
    end

    %% ========================================
    %% Step 10: Sequence Parallel Gather (both paths)
    %% ========================================
    subgraph SP_Gather ["_gather_from_sequence_parallel  [both paths]"]
        direction TB
        SPG_Check{"config.sequence_parallel?"}:::main_node
        SPG_Gather["gather_from_sequence_parallel_region<br/>(all-gather along token dim)"]:::main_node
        SPG_DePad["Remove SP padding<br/>x[:total_tokens]"]:::main_node
        SPG_Skip["Pass through"]:::main_node

        SPG_Check -- Yes --> SPG_Gather --> SPG_DePad
        SPG_Check -- No --> SPG_Skip
    end
    Hidden2 --> SPG_Check
    SPG_DePad --> Gather_Out(("hidden_states<br/>Shape: [s, 1, h]")):::io_node
    SPG_Skip --> Gather_Out

    %% ========================================
    %% Step 11: Remove Sequence Dimension
    %% ========================================
    Squeeze["Squeeze dim=1<br/>[s, 1, h] → [s, h]"]:::main_node
    Gather_Out --> Squeeze

    %% ========================================
    %% Output
    %% ========================================
    Squeeze --> FinalOutput(("Output<br/>Shape: [total_patches, h]")):::io_node
```

## Architecture Notes

| Component | Description |
|---|---|
| **_SCATTER_BEFORE_PATCH_EMBED** | Module-level flag read from env var `SCATTER_BEFORE_PATCH_EMBED` at import time. `=1`: scatter x **before** patch_embed (each rank processes s/tp patches). `=0` (default): patch_embed on full sequence, scatter after. Cannot use `=1` with `ParallelPatchEmbed` |
| **PatchEmbed** | Converts raw pixel patches `[*, C*P*P]` into embeddings `[*, h]`. Three variants: `ParallelPatchEmbed` (TP ColumnParallelLinear, SCATTER=0 only), `TorchLinearPatchEmbed` (plain nn.Linear), `PatchEmbed` (Conv2d) |
| **VideoRotaryEmbeddingSplit466** | 3D RoPE with **4:6:6** dimension split for Temporal / Height / Width. `unit = head_dim // 32`. Always computed on the **full** (replicated) sequence regardless of SCATTER path |
| **convert_rope_to_block_layout** | Reorders RoPE from row-major to 2x2 spatial-merge block order when `spatial_merge_size=2` |
| **pre_layernorm** | TENorm (LayerNorm or RMSNorm, eps=1e-4) applied **before** the transformer stack, on the local scattered shard |
| **TransformerBlock** | N layers of `TransformerLayer` (Self-Attention + MLP). Uses **TELayerNormColumnParallelLinear** (fused LN + Linear) for QKV and FC1, **TEDotProductAttention** for FlashAttention, **TERowParallelLinear** for projection and FC2 |
| **Sequence Parallel** | Token-dimension scatter/gather around transformer. Padding ensures divisibility by `tp_size`. RoPE and cu_seqlens stay **replicated** on all ranks in both paths |
| **Packed Sequences** | Variable-length samples packed together; `cu_seqlens` drives FlashAttention's `thd` format |

### SCATTER_BEFORE_PATCH_EMBED Path Comparison

| Stage | SCATTER=0 (default) | SCATTER=1 (optimised) |
|---|---|---|
| Before patch_embed | x replicated `[s, C*P*P]` | x scattered `[s/tp, C*P*P]` |
| patch_embed input | full sequence | 1/tp of sequence per rank |
| patch_embed compatible types | all three variants | `TorchLinearPatchEmbed`, `PatchEmbed` only |
| After patch_embed, before unsqueeze | `[s, h]` | `[s/tp, h]` |
| SP pad + scatter x | in `_scatter_for_sequence_parallel` | already done before patch_embed |
| RoPE + cu_seqlens padding | in `_scatter_for_sequence_parallel` | in post-embed pad step |
| Transformer input | `[s/tp, 1, h]` | `[s/tp, 1, h]` (same) |
| Gather | `_gather_from_sequence_parallel` (same) | same |

### Default VisionConfig (ViT-L/14)

| Parameter | Value |
|---|---|
| `num_layers` | 24 |
| `hidden_size` (h) | 1024 |
| `ffn_hidden_size` (4h) | 4096 |
| `num_attention_heads` | 16 |
| `kv_channels` | 64 |
| `patch_size` (P) | 14 |
| `in_channels` (C) | 3 |
| `spatial_merge_size` | 2 |
| `frame_windows_size` | 4 |
| `normalization` | LayerNorm |
| `activation` | GeLU |
