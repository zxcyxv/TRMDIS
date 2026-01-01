# train_film_smoe2.py — Core Training Mechanics (Concise)

This note documents only the **core mechanisms** in `train_film_smoe2.py`:
FiLM, Router inputs, expert mixing, and the exact losses that shape them.

## 1) FiLM: what it is and where it is applied

**Mechanism**

FiLM modulates Transformer features using scale/shift generated from the
last position:

```
h_out = γ(pos) ⊙ RMSNorm(h) + β(pos)
```

**How γ/β are generated**

`SpatialConditioningNetwork` takes only `(start_x_7, start_y_7)`:

1) Normalize by `[105, 68]`
2) Fourier encode with 32 frequencies → 64-dim `[cos, sin]`
3) Concatenate `[pos_norm, fourier]` → 66-dim
4) MLP → `γ`, `β` (each size = `d_model`)

**Where FiLM is applied**

After each Transformer block:

```
for i in layers:
    x = TransformerBlock(x)
    x = FiLMLayer(x, γ, β)
```

So the **same** `(γ, β)` modulates **all layers** and **all time steps**.

## 2) Router: exact inputs and Fourier augmentation

**Base input (6 dims)** to the router:

- `pos_norm = (start_x_7, start_y_7) / [105, 68]`
- boundary distances:
  - `x/105`, `(105-x)/105`, `y/68`, `(68-y)/68`

So the raw router vector is:

```
[x_norm, y_norm, dist_left, dist_right, dist_bottom, dist_top]
```

**Fourier expansion**

A fixed random matrix `B ∈ R^{6×32}` projects the 6D vector to 32
frequencies. Router input becomes:

```
[ raw_6,
  cos(2π Bx),
  sin(2π Bx) ]  =>  6 + 64 = 70 dims
```

**Gate output**

```
logit = MLP(fourier_input) / temperature
gate  = sigmoid(logit)
```

- `gate ≈ 0` → In-field expert
- `gate ≈ 1` → Boundary expert

## 3) Experts and mixing (heteroscedastic head)

Each expert predicts three vectors (per sample):

- `μ` (mean) : 2D
- `log_var`  : 2D
- `residual` : 2D

These are mixed by the router:

```
μ      = gate * μ_B + (1-gate) * μ_A
logvar = gate * lv_B + (1-gate) * lv_A
res    = gate * r_B  + (1-gate) * r_A
```

Final output:

```
σ = exp(0.5 * logvar)
y_final = μ + σ ⊙ res
```

## 4) Auxiliary head (what it outputs)

From the CLS token:

- `aux_dist`: distance-to-boundary regression
- `aux_zone`: 4-class logits (In-field / Top / Bottom / Goal-line)

These logits are also exported later as XGBoost features.

## 5) Losses that drive the router and experts

Total loss (with weights from `TrainingScheduler`):

1) **Gaussian NLL** on `μ, log_var`
2) **MSE** on `y_final`
3) **Aux distance** (`smooth_l1`) on `aux_dist`
4) **Aux zone** (`cross_entropy`, 4-class) on `aux_zone`
5) **Gate supervision** (`BCE`) with target:
   - `zone == 0` → 0
   - `zone in {1,2,3}` → 1

This gate loss is the **direct supervision** that forces the router to
separate in-field vs boundary.

## 6) Temperature schedule (routing sharpness)

`TrainingScheduler.get_temperature()`:

- Warmup: `T = 2.0`
- Anneal: linear to `T = 0.5`
- Late: `T = 0.5`

Validation is run at **fixed `T = 0.05`** to force hard routing.

## 7) What drives performance in practice

- **FiLM** aligns the Transformer representation with the last position.
- **Fourier router input** makes the gate highly sensitive to boundary
  micro-structure (goal line vs near line).
- **Gate supervision** turns the router into a true spatial classifier.
- **Aux zone logits** give the model explicit 4-zone geometry.

