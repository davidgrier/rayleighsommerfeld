;+
; NAME:
;     gpu_rayleighsommerfeld
;
; PURPOSE:
;     Computes Rayleigh-Sommerfeld back-propagation of
;     a normalized hologram of the type measured by
;     digital video microscopy.  Uses GPUlib routines
;     for hardware acceleration.
;
; CATEGORY:
;     Holographic microscopy
;
; CALLING SEQUENCE:
;     b = gpu_rayleighsommerfeld(a,z)
;
; INPUTS:
;     a: hologram recorded as image data normalized
;         by a background image. 
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
;     hanning: If set, use Hanning window to suppress
;         Gibbs phenomeon.
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
;
; EXAMPLES:
; Visualize the intensity as a function of height z above the image
; plane given a normalized holographic image a.

; IDL> lambda = 0.6328 / 1.336 ; HeNe in water
; IDL> for z = 1., 200 do begin $
; IDL>    tvscl, abs(rayleighsommerfeld(a, z, lambda=lambda))^2 & $
; IDL> endfor
;
; Visualize the phase singularity due to a particle located along the
; line y = 230 in the image a, assuming that the particle is no more
; than 200 pixels above the image plane.
;
; IDL> help, a
; A     FLOAT     = Array[640, 480]
; IDL> phi = fltarr(640,200)
; IDL> for z = 1., 200 do begin $
; IDL> phiz = atan(rayleighsommerfeld(a,z,lambda=lambda),/phase) & $
; IDL> phi[*,z-1] = phiz[*,230] & $
; IDL> endfor
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
; 06/10/2010: DGG. Documentation fixes.  Added COMPILE_OPT.
; 07/27/2010: DGG. Added HANNING.
; 10/20/2010: DGG: Subtract 1 from (normalized) hologram
;   rather than expecting user to do subtraction.
;
; Copyright (c) 2006-2010 Sanghyuk Lee and David G. Grier
;-
function gpu_crayleighsommerfeld, a_, z, $
                                  lambda = _lambda, $ ; wavelength of light
                                  mpp = mpp, $        ; micrometers per pixel
                                  nozphase = nozphase, $
                                  double = double

COMPILE_OPT IDL2

type = (keyword_set(double) &&  gpudoublecapable()) ? 9 : 6 ; dcomplex or complex

; hologram dimensions
sz = size(a_)
ndim = sz[0]
if ndim gt 2 then message, "requires two-dimensional hologram"
nx = float(sz[1])
ny = float(sz[2])

a = gpuPutArr(a_)
a = gpuFix(a, TYPE = type, /OVERWRITE)

; volumetric slices
nz = n_elements(z)              ; number of z planes

; parameters
if n_elements(_lambda) ne 1 then $
   _lambda = 0.632              ; HeNe laser

if n_elements(mpp) ne 1 then $
   mpp = 0.135                  ; Nikon rig

lambda = _lambda/mpp                      ; wavelength in pixels
ik = complex(0., 2. * !pi / lambda)       ; wavenumber in radians/pixel

; phase factor for Rayleigh-Sommerfeld propagator in Fourier space
; Refs. [2] and [3]
qx = gpuMake_Array(nx, ny, /INDEX, TYPE = type)
qy = gpuFloor(1., 1./nx, qx, 0., 0.)
qx = gpuSub(1./nx, qx, 1., qy, -0.5, LHS = qx)
qy = gpuAdd(1./ny, qy, 0., qy, -0.5, LHS = qy)
qx = gpuMult(qx, qx, LHS = qx, /NONBLOCKING)
qy = gpuMult(qy, qy, LHS = qy)
qx = gpuAdd(-lambda^2, qx, -lambda^2, qy, 1., LHS = qx) ; qx = 1 - q^2
qx = gpuSqrt(qx, LHS = qx)

w = round(float(nx)/2.)         ; width of block to move
gpuSubArr, qx, [0, w-1], -1, qy, [nx-w, -1], -1
gpuSubArr, qx, [w, -1], -1, qy,  [0, nx-w-1], -1
h = round(float(ny)/2.)         ; height of block to move
gpuSubArr, qy, -1, [0, h-1], qx, -1, [ny-h, -1]
gpuSubArr, qy, -1, [h, -1], qx, -1, [0, ny-h-1]
; NOTE: Bug in gpuShift for GPUlib 1.4.2
;qx = gpuShift(qx, long(nx/2.), long(ny/2.), LHS = qx)

stop

res = complexarr(nx, ny, nz, /NOZERO)

plan = 0UL
a = gpuAdd(1., a,  0., a, -1., LHS = a)
a = gpuFFT(a, PLAN = plan, LHS = a)

Hqz = gpuMake_Array(nx, ny, TYPE = type, /NOZERO)
for j = 0, nz-1 do begin
;   if z[j] gt 0 then $
;      Hqz = exp(float(z[j]) * conj(qfactor)) $
;   else $
;      Hqz = exp(float(z[j]) * qfactor)

; Convolve field with Rayleigh-Sommerfeld propagator
   thisE = E * Hqz                            ; Convolve with propagator
   thisE = fft(thisE, 1, /center, /overwrite) ; Transform back to real space

   res[*,*,j] = thisE
endfor

return, res

end
