# Forward-mode automatic differentiation for high-order PDE derivatives

**Citations:**
- Cho et al., SPINN, NeurIPS 2023, arXiv:2306.15969 (forward mode with separable networks).
- Shi, Hu, Lin & Kawaguchi, STDE, NeurIPS 2024, arXiv:2412.00088 (Taylor-mode jets).
- Dangel, Müller & Zeinhofer, KFAC for PINNs, NeurIPS 2024, arXiv:2405.15603 (Taylor mode).

**Mechanism.** For a pointwise model u(x,t), d^k u/dx^k for every point is computed by k nested JVPs with a
tangent of ones. Parameter gradients follow by reverse mode through the JVP graph (forward-over-reverse).

**Implementation.** `src/physref/derivatives.py` (`derivs_forward`, `_nested_jvp`) with `torch.func.jvp`.
Values and parameter gradients equal reverse mode to round-off (tests).

**Deviations.** This is nested first-order forward mode, **not** Taylor mode. PyTorch has no native
Taylor-mode jets.

**Computational implications (measured, B2-PROF-001).** **4.9× slower** than nested reverse mode for the
strong residual (750 vs 154 ms per step), because each nesting level re-traces the network.

**Conclusion.** Not worth pursuing in PyTorch for this architecture. Revisit with JAX `jet` or separable
structure.

**Novelty status.** ESTABLISHED.
