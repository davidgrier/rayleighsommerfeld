;+
; NAME:
;     gpu_rayleighsommerfeld
;
; PURPOSE:
;     Computes Rayleigh-Sommerfeld back-propagation of
;     a normalized hologram of the type measured by
;     digital video microscopy. Uses GPUlib for
;     hardware acceleration.
;
; CATEGORY:
;     Holographic microscopy
;
; CALLING SEQUENCE:
;     b = gpu_rayleighsommerfeld(a,z)
;
; INPUTS:
;     a: two-dimensional image data 
;     z: displacement from the focal plane [pixels]
;         If z is an array of displacements, the field
;         is computed at each plane.
;
; KEYWORDS:
;     lambda: Wavelength of light in medium [micrometers]
;         Default: 0.632 -- HeNe in air
;     mpp: Micrometers per pixel
;         Default: 0.135
;
; KEYWORD_FLAGS:
;     nozphase: By default, a phase factor of kz is removed 
;         from the propagator to eliminate the (distracting)
;         axial phase gradient.  Setting this flag leaves 
;         this factor (and the axial phase ramp) in.
;
; OUTPUTS:
;     b: complex field at height z above image plane,
;         computed by convolution with
;         the Rayleigh-Sommerfeld propagator.
;
; REFERENCES:
; 1. S. H. Lee and D. G. Grier, 
;   "Holographic microscopy of holographically trapped 
;   three-dimensional structures,"
;   Optics Express, 15, 1505-1512 (2007).
;
; 2. J. W. Goodman, "Introduction to Fourier Optics,"
;    (McGraw-Hill, New York 2005).
;
; 3. G. C. Sherman,
;   "Application of the convolution theory to Rayleigh's integral
;   formulas,"
;   Journal of the Optical Society of America 57, 546-547 (1967).
;
; 4. P. Messmer, P. J. Mullowney and B. E. Granger,
;    "GPULib: GPU computing in high-level languages,"
;    Computer Science and Engineering 10, 70-73 (2008).
;
; PROCEDURE:
;     Convolution with Rayleigh-Sommerfeld propagator using
;     Fourier convolution theorem.
;     Calls GPUlib routines for hardware-accelerated calculations.
;
; EXAMPLES:
; Visualize the intensity as a function of height z above the image
; plane given a normalized holographic image a.

; IDL> lambda = 0.6328 / 1.336 ; HeNe in water
; IDL> b = gpu_rayleighsommerfeld(a, findgen(200)+1., lambda=lambda)
; IDL> for z = 0, 199 do tvscl, abs(b[*,*,z])^2
;
; Visualize the phase singularity due to a particle located along the
; line y = 230 in the image a, assuming that the particle is no more
; than 200 pixels above the image plane.
;
; IDL> help, a
; A     FLOAT     = Array[640, 480]
; IDL> phi = fltarr(640,200)
; IDL> z = findgen(200) + 1.
; IDL> b = gpu_rayleighsommerfeld(a,z)
; IDL> for z = 0, 199 do phi[*,z] = atan(b[*,230,z],/phase)
; IDL> tvscl, phi
;
; MODIFICATION HISTORY:
; 11/28/2006: Sang-Hyuk Lee, New York University, 
;   Original version (called FRESNELDHM.PRO).
;   Based on fresnel.pro by D. G. Grier
; 10/21/2008: David G. Grier, NYU.  Adapted from FRESNELDHM
;   to provide more uniform interface.
; 11/07/2009: DGG.  Code and documentation clean-up.  Properly
;   handle places where qsq > 1.  Implemented NOZPHASE.
; 11/09/2009: DGG. Initial implementation of simultaneous projection
;   to multiple z planes for more efficient volumetric
;   reconstructions.  Eliminate matrix reorganizations.
; 11/09/2009: DGG. Derived from RAYLEIGHSOMMERFELD.PRO.
;   Initial GPUlib implementation.  Documentation updated accordingly.
; 11/12/2009: DGG. Simplified shifting code.  Fixed GPU memory leak.
;   Documentation fixes.
; 02/24/2010: DGG. Cleaned up another GPU memory leak.  Added hooks
;   for error tracking.  Defined CUFFT plan for improved efficiency.
; 06/10/2010: DGG. Documentation fixes.  Added COMPILE_OPT.
; 11/18/2010: DGG. Subtract 1 from input hologram, rather than
;   expecting the user to subtract 1 first.  This maintains
;   consistency with RAYLEIGHSOMMERFELD.
;
; Copyright (c) 2006-2010 Sanghyuk Lee and David G. Grier.
;-
function gpu_rayleighsommerfeld, _a, z, $
                                 lambda = _lambda, $ ; wavelength of light
                                 mpp = mpp, $        ; micrometers per pixel
                                 nozphase = nozphase

COMPILE_OPT IDL2

; hologram dimensions
sz = size(_a)
ndim = sz[0]
if ndim ne 2 then message, "requires two-dimensional hologram"
nx = float(sz[1])
ny = float(sz[2])

; volumetric slices
nz = n_elements(z)              ; number of z planes

; parameters
if n_elements(_lambda) ne 1 then $
   _lambda = 0.632              ; HeNe laser
if n_elements(mpp) ne 1 then $
   mpp = 0.135                  ; Nikon rig

lambda = _lambda/mpp            ; wavelength in pixels
k = 2.*!pi/lambda               ; wavenumber in radians/pixel

; phase factor for Rayleigh-Sommerfeld propagator in Fourier space
; Refs. [2] and [3]
qx = gpuFindgen(nx, ny)
qy = gpuFltArr(nx, ny)
qy = gpuFloor(1., 1./nx, qx, 0., 0., LHS = qy)
qx = gpuSub(1./nx, qx, 1., qy, -0.5, LHS = qx)
qy = gpuAdd(1./ny, qy, 0., qy, -0.5, LHS = qy)
qx = gpuMult(qx, qx, LHS = qx, /NONBLOCKING)
qy = gpuMult(qy, qy, LHS = qy)
qx = gpuAdd(-lambda^2, qx, -lambda^2, qy, 1., LHS = qx)
; qx = 1 - q^2

; we have to shift this for consistency with gpuFFT
;w = round(float(nx)/2.)         ; width of block to move
;gpuSubArr, qx, [0, w-1], -1, qy, [nx-w, -1], -1
;gpuSubArr, qx, [w, -1], -1, qy,  [0, nx-w-1], -1
;h = round(float(ny)/2.)         ; height of block to move
;gpuSubArr, qy, -1, [0, h-1], qx, -1, [ny-h, -1]
;gpuSubArr, qy, -1, [h, -1], qx, -1, [0, ny-h-1]
; NOTE: Bug in gpuShift for GPUlib 1.4.2, fixed in 1.4.4
qx = gpuShift(qx, round(nx/2.), round(ny/2.), LHS = qx)

; 1 - q^2 is negative for q > 1.  Find out where ...
nmask = gpuLT(1., qx, 0., qx, 0., /NONBLOCKING)
pmask = gpuAdd(-1., nmask, 0., nmask, 1., /NONBLOCKING)
qx = gpuAbs(qx, LHS = qx)
qx = gpuSqrt(k, 1., qx,  0., 0., LHS = qx)
qy = gpuMult(nmask, qx, LHS = qy)
qx = gpuMult(1., pmask, 1., qx, keyword_set(nozphase) ? 0. : -k, LHS = qx)
; qx = k Re{ Sqrt(1 - q^2) } - k
; qy = k Im{ Sqrt(1 - q^2) }

; Perform convolution
ReHqz = gpuFltArr(nx, ny, /NOZERO)
ImHqz = gpuFltArr(nx, ny, /NOZERO)
aHqz = gpuComplexarr(nx, ny, /NOZERO)
expqz = gpuFltarr(nx, ny, /NOZERO)

res = complexarr(nx,ny,nz)

; Convolve hologram with RS propagator
plan = 0UL
a = gpuPutArr(_a)
a = gpuAdd(1., a, 0., a, -1., LHS = a)
a = gpuFFT(a, PLAN = plan, LHS = a)

for j = 0, nz-1 do begin
   ; H(q,z) = exp(-i kz sqrt(1 - q^2))
   expqz = gpuExp(1., -abs(z[j]), qy, 0., 0., LHS = expqz, /NONBLOCKING)
   ReHqz = gpuCos(1., z[j], qx, 0., 0., LHS = ReHqz)
   ReHqz = gpuMult(expqz, ReHqz, LHS = ReHqz, /NONBLOCKING)
   ImHqz = gpuSin(-1., z[j], qx, 0., 0., LHS = ImHqz)
   ImHqz = gpuMult(expqz, ImHqz, LHS = ImHqz)
   aHqz = gpuComplex(ReHqz, ImHqz, LHS = aHqz)
   aHqz = gpuMult(a, aHqz, LHS = aHqz)
   aHqz = gpuFFT(aHqz, /INVERSE, PLAN = plan, LHS = aHqz)
   res[*, *, j] = gpuGetArr(aHqz)
endfor

gpuFFT, PLAN = plan, /DESTROYPLAN

; Explicit freeing of gpu variables no longer required under
; GPUlib 1.4.0
; gpuFree, [a, qx, qy, nmask, pmask, ReHqz, ImHqz, aHqz, expqz]

return, res

end
