# Scaling on Jean Zay H100 (gpu_p6)

100 warmup steps discarded; step time is the **mean** of the rest, since node-hours are mean x steps. Epoch = 64,000 samples.

### DeltaTok tokenizer — strong (global batch 32) and weak scaling

| ladder | GPUs | nodes | Total Batch Size | Batch/GPU | Training Time/Epoch (s) | Speed-up (vs. 4 GPUs) | Efficiency (%) |
|---|---|---|---|---|---|---|---|
| strong | 4 | 1 | 32 | 2 | 3670 | 1.00 | 100 |
| strong | 8 | 2 | 32 | 2 | 1920 | 1.91 | 96 |
| **strong** | **16** | **4** | **32** | **2** | **1033** | **3.55** | **89** |
| strong | 32 | 8 | 32 | 1 | 740 | 4.96 | 62 |
| weak | 4 | 1 | 8 | 2 | 3779 | 1.00 | 100 |
| weak | 8 | 2 | 16 | 2 | 1989 | 1.90 | 95 |
| **weak** | **16** | **4** | **32** | **2** | **1033** | **3.66** | **91** |
| weak | 32 | 8 | 64 | 2 | 554 | 6.83 | 85 |

### DeltaTok-flow world model — strong (global batch 128) and weak scaling

| ladder | GPUs | nodes | Total Batch Size | Batch/GPU | Training Time/Epoch (s) | Speed-up (vs. 4 GPUs) | Efficiency (%) |
|---|---|---|---|---|---|---|---|
| strong | 4 | 1 | 128 | 16 | 872 | 1.00 | 100 |
| strong | 8 | 2 | 128 | 16 | 449 | 1.94 | 97 |
| strong | 16 | 4 | 128 | 8 | 260 | 3.35 | 84 |
| **strong** | **32** | **8** | **128** | **4** | **183** | **4.76** | **60** |
| weak | 4 | 1 | 64 | 16 | 867 | 1.00 | 100 |
| weak | 8 | 2 | 128 | 16 | 449 | 1.93 | 97 |
| weak | 16 | 4 | 256 | 16 | 226 | 3.83 | 96 |
| **weak** | **32** | **8** | **512** | **16** | **113** | **7.65** | **96** |

### OccAny geometry foundation model — strong (global batch 32) and weak scaling

| ladder | GPUs | nodes | Total Batch Size | Batch/GPU | Training Time/Epoch (s) | Speed-up (vs. 4 GPUs) | Efficiency (%) |
|---|---|---|---|---|---|---|---|
| strong | 4 | 1 | 32 | 2 | 5198 | 1.00 | 100 |
| strong | 8 | 2 | 32 | 2 | 2703 | 1.92 | 96 |
| **strong** | **16** | **4** | **32** | **2** | **1354** | **3.84** | **96** |
| strong | 32 | 8 | 32 | 1 | 815 | 6.38 | 80 |
| weak | 4 | 1 | 8 | 2 | 5253 | 1.00 | 100 |
| weak | 8 | 2 | 16 | 2 | 2707 | 1.94 | 97 |
| weak | 16 | 4 | 32 | 2 | 1354 | 3.88 | 97 |
| **weak** | **32** | **8** | **64** | **2** | **682** | **7.70** | **96** |

Bold = the production job size from the milestone table. Strong holds the total batch fixed; weak holds batch/GPU fixed. Time per epoch assumes an epoch of 64,000 samples, which is what lets speed-up and efficiency be quoted for both ladders on the same footing.