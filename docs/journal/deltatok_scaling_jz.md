# DeltaTok scaling on Jean Zay H100 (gpu_p6)

100 warmup steps discarded; step time is the **mean** of the rest, since node-hours are mean x steps. Epoch = 64,000 samples.

### DeltaTok tokenizer — strong (global batch 32) and weak scaling

| ladder | GPUs | Total Batch Size | Batch/GPU | Training Time/Epoch (s) | Speedup (vs. 4 GPU) | Efficiency (%) |
|---|---|---|---|---|---|---|
| strong | 4 | 32 | 2 | 3670 | 1.00 | 100 |
| strong | 8 | 32 | 2 | 1920 | 1.91 | 96 |
| **strong** | **16** | **32** | **2** | **1033** | **3.55** | **89** |
| strong | 32 | 32 | 1 | 740 | 4.96 | 62 |
| weak | 4 | 8 | 2 | 3779 | 1.00 | 100 |
| weak | 8 | 16 | 2 | 1989 | 1.90 | 95 |
| **weak** | **16** | **32** | **2** | **1033** | **3.66** | **91** |
| weak | 32 | 64 | 2 | 554 | 6.83 | 85 |

### DeltaTok-flow world model — strong (global batch 128) and weak scaling

| ladder | GPUs | Total Batch Size | Batch/GPU | Training Time/Epoch (s) | Speedup (vs. 4 GPU) | Efficiency (%) |
|---|---|---|---|---|---|---|
| strong | 4 | 128 | 16 | 872 | 1.00 | 100 |
| strong | 8 | 128 | 16 | 449 | 1.94 | 97 |
| strong | 16 | 128 | 8 | 260 | 3.35 | 84 |
| **strong** | **32** | **128** | **4** | **183** | **4.76** | **60** |
| weak | 4 | 64 | 16 | 867 | 1.00 | 100 |
| weak | 8 | 128 | 16 | 449 | 1.93 | 97 |
| weak | 16 | 256 | 16 | 226 | 3.83 | 96 |
| **weak** | **32** | **512** | **16** | **113** | **7.65** | **96** |

Bold = the production job size from the milestone table. Strong holds the total batch fixed; weak holds batch/GPU fixed. Time per epoch assumes an epoch of 64,000 samples, which is what lets speed-up and efficiency be quoted for both ladders on the same footing.