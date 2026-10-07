# Adam with exponential learning-rate decay (part of B1)

**Citations:**
- Kingma & Ba, ICLR 2015, arXiv:1412.6980.
- Schedule from the reference code of the target paper's lineage (MultiscalePINNs):
  `tf.train.exponential_decay(1e-3, step, 1000, 0.9)`.

**Mechanism.** lr(s) = 1e-3 · 0.9^(s/1000), continuous (not staircase).

**Implementation.** `src/beampinn/optimization/optimizers.py` (Batch 1). Verified values: 1e-3, 9e-4,
8.1e-4, 5.9e-4 at steps 0, 1k, 2k, 5k.

**Deviations.** None from the reference code.

**Computational implications.** None beyond Adam.

**Batch-1 outcome (Y phase).**
- Improved the hard-constrained PINN (Y1).
- Insufficient alone.
- By 20k steps the LR had decayed to 1.2e-4. This is a possible contributor to the decelerating front;
  untested.

**Novelty status.** ESTABLISHED.
