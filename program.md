# Autoresearch Program

## Goal
Optimize `melon_simerr_sum` (lower is better) by modifying `models/models_gru.py`.

## Ideas to Explore
Primary evaluation is `test/test_gru.py` on melon with horizon 50. Record all rows in
`melon_summary.csv`; the autoresearch scalar is the sum of the four `simerr` rows:
`simerr_pos + simerr_vel + simerr_rot + simerr_omega`.

Baseline:
- Commit: 2e25c29eba20d7f7eeaf2e729f1f868111c957c9
- Checkpoint: `out/models/raw_gru_random_square_chirp_bs4096_snapshot.pt`
- Summary: `out/predictions/raw_gru_random_square_chirp_bs4096_snapshot_model_multistep/melon_summary.csv`
- Metric: 41.7740039233905
- Simerr: pos=2.363310, vel=9.201048, rot=5.205946, omega=25.003700

Candidate experiments:
1. Hidden-state normalization: apply parameter-free layer normalization to GRU hidden
   features before they are reused as GRU/residual features. Keep residual input shape
   `12 + 4 + H` unchanged.
2. DIEN-style causal hidden attention: maintain a causal history of hidden states,
   compute attention from current state/control/hidden query to historical hidden
   states, and gate-mix the context with the current hidden state. Do not concatenate
   context into residual input, because train/test construct residual input as
   `4 + gru_hidden_dim`.
3. GRU input simplification: remove hidden from the explicit GRU input and let GRUCell
   receive hidden only through its recurrent state; keep residual input unchanged.

## Constraints
- Time budget per experiment: 8h
- Only modify: models/models_gru.py
- Formal runs happen on remote host `4060` under tmux, using
  `/home/ubuntu/miniconda3/envs/dynamics_learning/bin/python`.
