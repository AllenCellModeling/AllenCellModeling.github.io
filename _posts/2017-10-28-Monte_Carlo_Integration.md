---
title: Monte Carlo Integration
summary: Working out a variation metric for the IC using monte carlo integration of a toy problem
---

# Currently an unfinished rough draft

# toy 2D → 3D problem


```python
import numpy as np
import numba
```

### exact solution

Our map is a simple one, from 2D $\boldsymbol{z}$ space to 3D $\boldsymbol{x}$ space.  We endow the 2D latent space with a gaussian density, i.e
\begin{align}
\rho(\boldsymbol{z}) &= \frac{1}{2\pi} e^{-\tfrac{1}{2} |\boldsymbol{z}|^2 }
\end{align}
As a toy map, take
\begin{align}
\boldsymbol{z} &= (z_1, z_2) \\
\boldsymbol{x}(\boldsymbol{z}) &= \tfrac{1}{\sqrt[4]{2}} \left(\tfrac{1}{\sqrt{2}}z_1^2, \tfrac{1}{\sqrt{2}}z_2^2, z_1 z_2 \right)
\end{align}
The Jacobian of this transformation is
\begin{align}
J(\boldsymbol{x}) &= \frac{d \boldsymbol{x}}{d \boldsymbol{z}} \\
                  &= \sqrt[4]{2} \pmatrix{          z_1          &             0       \cr
                                                     0           &            z_2      \cr
                                          \tfrac{1}{\sqrt{2}}z_2 & \tfrac{1}{\sqrt{2}}z_1 }
\end{align}
A scalar measure Jacobian we can use is $D(\boldsymbol{z}) = \sqrt{ \det{ \left( J(\boldsymbol{z})^T J(\boldsymbol{z}) \right)} }$ (see https://en.wikipedia.org/w/index.php?title=Determinant).

\begin{align}
J(\boldsymbol{x})^T J(\boldsymbol{x}) &= \sqrt{2}
                                         \pmatrix{z_1 &  0  & \tfrac{1}{\sqrt{2}}z_2 \cr
                                                   0  & z_2 & \tfrac{1}{\sqrt{2}}z_1}
                                         \pmatrix{z_1 &  0  \cr 0  & z_2 \cr
                                                  \tfrac{1}{\sqrt{2}}z_2 & \tfrac{1}{\sqrt{2}}z_1} \\
                                      &= \pmatrix{\sqrt{2}z_1^2 + \tfrac{1}{\sqrt{2}}z_2^2 & \tfrac{1}{\sqrt{2}} z_1 z_2 \cr
                                                     \tfrac{1}{\sqrt{2}} z_1 z_2   & \tfrac{1}{\sqrt{2}}z_1^2 + \sqrt{2} z_2^2 }
\end{align}
Taking the determinant we get
\begin{align}
\det{ \left( J(\boldsymbol{z})^T J(\boldsymbol{z}) \right)} &= z_1^4 + 2 z_1^2 z_2^2 + z_2^4 \\
                                                            &= \left( z_1^2 + z_2^2 \right)^2
\end{align}
Our measure is then simply
\begin{align}
D(\boldsymbol{z}) &= \sqrt{ \det{ \left( J(\boldsymbol{z})^T J(\boldsymbol{z}) \right)} } \\
                  &= \sqrt{\left( z_1^2 + z_2^2 \right)^2} \\
                  &= z_1^2 + z_2^2
\end{align}

To find the expectation of this measure over the entire $\boldsymbol{z}$ space, we integrate over the space, weighting by the density of $\boldsymbol{z}$ in that space:
\begin{align}
\Bbb E [ D(\boldsymbol{z}) ] &= \int D(\boldsymbol{z}) \rho(\boldsymbol{z}) d\boldsymbol{z} \\
                             &= \int_{-\infty}^\infty \int_{-\infty}^\infty \left( z_1^2 + z_2^2 \right)
                                \frac{1}{2\pi} e^{-\tfrac{1}{2} (z_1^2 + z_2^2)^2 } d z_1 d z_2 \\
                             &= \int_{0}^{2\pi} \int_{0}^\infty \frac{1}{2\pi} e^{ -\tfrac{1}{2} r^2 } r^2 r dr d\theta \\
                             &= \int_{0}^\infty e^{ -\tfrac{1}{2} r^2 } r^3 r dr \\
                             &= 2
\end{align}

### numerical solution


```python
@numba.jit(nopython=True)
def x(z):
    z1,z2 = z
    return np.array( [z1**2/np.sqrt(2.0), z2**2/np.sqrt(2.0), z1*z2] ) / np.sqrt(np.sqrt(2.0))
```


```python
@numba.jit(nopython=True)
def sqrt_det_jacobian(z, delta=1e-6):
    J = np.zeros((len(x(z)),len(z)))
    for i,z_i in enumerate(z):
        z_delta = z.copy()
        z_delta[i] += delta
        J[:,i] = (x(z_delta)-x(z))/delta
    det_JtJ = np.linalg.det( np.dot(J.transpose(), J) )
    return np.sqrt(det_JtJ)
```


```python
z_points = np.random.randn(1024,2)
jac_samples = np.zeros(len(z_points))
for i,zi in enumerate(z_points):
    jac_samples[i] = sqrt_det_jacobian(zi,delta=1e-9)
np.mean(jac_samples)
```




    2.0324136516733455



### notes re pytorch

https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059

https://discuss.pytorch.org/t/more-efficient-implementation-of-jacobian-matrix-computation/6960


```python

```
