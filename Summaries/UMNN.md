## Unconstrained Monotonic Neural Networks - UMNN

- The authors insight is that a function is monotonic as long as its derivative remains of constant sign. Therefore, they introduce a new reversible scalar transformation defined via a free-form neural network whose only constraint is the positiveness of its output. 
	- This architecture guarantess monoticity without constrains on the expressiveness of the hypothesis class, contrary to classical approaches. For example, architectures that ensure monotonicity typically enforce constraints on weight and activation functions, which enables invertibility but leads to a cap on the expressiveness of the resulting transformations.
	- This leads to multiple advantages: it enables the use of any state-of-the-art neural architecture, simplifies weight initialization, and leads to a more lightweight evaluation of the Jacobian.

### Take Away
- When combined with autoregressive flow (UMNN-MAF) for normalizing flows, they require fewer parameters than NAF for big datasets.
