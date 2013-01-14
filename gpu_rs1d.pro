;+
; NAME:
;    gpu_rs1d: NOTE Currently BROKEN!!!
;
; PURPOSE:
;    Computes Rayleigh-Sommerfeld back-propagation of
;    a normalized hologram along a specified axial line
;    using GPULib for hardware acceleration.
;
; CATEGORY:
;    Holographic microscopy
;
; CALLING SEQUENCE:
;    b = gpu_rs1d(a, z, rc)
;
; INPUTS:
;    a: hologram recorded as image data normalized
;        by a background image. 
;    z: displacement from the focal plane [pixels]
;        If z is an array of displacements, the field
;        is computed at rc for each height.
;    rc: [x,y] coordinates of center along which to compute
;        back-propagation [pixels]
;
; KEYWORDS:
;    lambda: Wavelength of light in medium [micrometers]
;        Default: 0.632 -- HeNe in air
;    mpp: Micrometers per pixel
;        Default: 0.135
;
; OUTPUTS:
;     b: complex field along line passing through rc in the plane
;        of the hologram.
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
;    Convolution with Rayleigh-Sommerfeld propagator using
;    Fourier convolution theorem.  Calls GPULib routines for
;    hardware-accelerated calculations.
;
; MODIFICATION HISTORY:
; 06/22/2012 Written by David G. Grier, New York University
; 06/24/2012 DGG GPU-accelerated version.
;
; Copyright (c) 2012 David G. Grier
;-
function gpu_rs1d, _a, z, rc, $
                   delta = delta, $
                   lambda = lambda, $ ; wavelength of light
                   mpp = mpp          ; micrometers per pixel
  
COMPILE_OPT IDL2

umsg = 'USAGE: b = gpu_rs1d(a, z, rc)'
if n_params() ne 3 then begin
   message, umsg, /inf
   return, -1
endif

; hologram
if ~isa(_a, /number, /array) then begin
   message, umsg, /inf
   message, 'a must be a numeric array', /inf
   return, -1
endif
sz = size(_a)
ndim = sz[0]
if ndim ne 2 then begin
   message, umsg, /inf
   message, 'a must be two-dimensional', /inf
   return, -1
endif

nx = float(sz[1])
ny = float(sz[2])

; axial samples
if ~isa(z, /number) then begin
   message, umsg, /inf
   message, 'z must be a numeric data type', /inf
   return, -1
endif
nz = n_elements(z)              ; number of z planes

if ~isa(rc, /number, /array) || n_elements(rc) ne 2 then begin
   message, umsg, /inf
   message, 'rc must be a two element numeric array', /inf
   return, -1
endif

; parameters
if ~isa(lambda, /number, /scalar) then $
   lambda = 0.632               ; HeNe laser in air
if ~isa(mpp, /number, /scalar) then $
   mpp = 0.135                  ; Nikon rig

ci = complex(0., 1.)
k = 2.*!pi*mpp/lambda           ; wavenumber in radians/pixel

; phase factor for Rayleigh-Sommerfeld propagator in Fourier space
; Refs. [2] and [3]
qx = gpufindgen(nx, ny)
qx = gpushift(qx, round(nx/2.), round(ny/2.), LHS = qx) ; shift for gpuFFT
qy = gpufltarr(nx, ny)
qrc = gpufltarr(nx, ny)
qfac = gpufltarr(nx, ny)

; coordinates of pixels
qy = gpufloor(1., 1./nx, qx, 0., 0., LHS = qy)
qx = gpuadd(1., qx, -nx, qy, 0., LHS = qx)
qx = gpuadd(1./nx, qx, 0., qx, -0.5, LHS = qx)
qy = gpuadd(1./ny, qy, 0., qy, -0.5, LHS = qy)

return, gpugetarr(qx)

qrc = gpuadd(k*rc[0], qx, k*rc[1], qy, 0., LHS = qrc) ; \vec{q} \cdot \vec{r}_c

qx = gpumult(qx, qx, LHS = qx, /NONBLOCKING)
qy = gpumult(qy, qy, LHS = qy)
qfac = gpuadd(-1., qx, -1, qy, 1., LHS = qfac)     ; 1 - (q/k)^2
qfac = gpusqrt(qfac, LHS = qfac)                   ; \sqrt(1 - (q/k)^2)
qfac = gpuadd(1., qfac, 0., qfac, -1., LHS = qfac) ; \sqrt(1 - (q/k)^2) - 1

a = gpuputarr(_a)
a = gpuadd(1., a, 0., a, -1., LHS = a)
b = gpufft(a)
ReFac = gpucos(qrc)
ImFac = gpusin(qrc)
fac = gpucomplex(ReFac, ImFac)
b = gpumult(b, fac, LHS = b)
res = complexarr(nz, /nozero)
for j = 0, nz-1 do begin
   ReFac = gpucos(1.,  k*z[j], qfac, 0., 0., LHS = ReFac)
   ImFac = gpusin(1., -k*z[j], qfac, 0., 0., LHS = ImFac)
   fac = gpucomplex(ReFac, Imfac, LHS = fac)
   fac = gpumult(b, fac, LHS = fac)
   res[j] = gputotal(fac)
endfor

return, res

end
