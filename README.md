# rayleighsommerfeld

**IDL routines for analyzing in-line holographic microscopy images
using Rayleigh-Sommerfeld back-propagation**

IDL is the Interactive Data Language, and is a product of
[Exelis Visual Information Solutions](http://www.exelisvis.com).

These routines are licensed under the GPLv3.

## What it does
* **rayleighsommerfeld**: Computes Rayleigh-Sommerfeld back-propagation of
a normalized hologram of the type measured by in-line digital video microscopy.

* **gpu_rayleighsommerfeld**: Hardware accelerated Rayleigh-Sommerfeld back-propagation
using GPULib.

* **rs1d**: Computes Rayleigh-Sommerfeld back-propagation of
a normalized hologram along a specified axial line.

* **gpu_rs1d**: Hardware-accelerated version of rs1d using GPULib.

* **rs_decon**: Implementation of Rayleigh-Sommerfeld deconvolution microscopy.

### References
1. S. H. Lee and D. G. Grier, 
"Holographic microscopy of holographically trapped 
three-dimensional structures,"
_Optics Express_ **15**, 1505-1512 (2007).

2. J. W. Goodman, "Introduction to Fourier Optics,"
(McGraw-Hill, New York 2005).

3. G. C. Sherman,
"Application of the convolution theory to Rayleigh's integral formulas,"
_Journal of the Optical Society of America_ **57**, 546-547 (1967).

4. P. Messmer, P. J. Mullowney and B. E. Granger,
"GPULib: GPU computing in high-level languages,"
_Computer Science and Engineering_ **10**, 70-73 (2008).

5. L. Dixon, F. C. Cheong and D. G. Grier,
"Holographic deconvolution microscopy for high-resolution particle tracking,"
_Optics Express_ **19**, 16410-16417 (2011).