; NAME:
;     rs_decon
;
; PURPOSE:
;     Computes the Rayleigh-Sommerfeld back-propagation 
;     of a normalized hologram, then deconvolves the volume 
;     with an optimized kernel.
;
; CATEGORY:
;     Holographic microscopy
;
; CALLING SEQUENCE:
;     b = rs_decon(a,z)
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
;     hanning: If set, use Hanning window to suppress
;         Gibbs phenomeon.
;
; OUTPUTS:
;     b: complex field at height z above image plane,
;         computed by convolution with
;         the Rayleigh-Sommerfeld propagator.
;
; REFERENCES:
; 1. L. Dixon, F. C. Cheong and D. G. Grier,
;   "Holographic deconvolution microscopy for high-resolution particle 
;   tracking,"
;   Optics Express 19, 16410-16417 (2011).
;
; 2. S. H. Lee and D. G. Grier, 
;   "Holographic microscopy of holographically trapped 
;   three-dimensional structures,"
;   Optics Express 15, 1505-1512 (2007).
;
; 3. J. W. Goodman, "Introduction to Fourier Optics,"
;    (McGraw-Hill, New York 2005).
;
; 4. G. C. Sherman,
;   "Application of the convolution theory to Rayleigh's integral
;   formulas,"
;   Journal of the Optical Society of America 57, 546-547 (1967).
;
; PROCEDURE:
;    Convolution with Rayleigh-Sommerfeld propagator using
;    Fourier convolution theorem followed by deconvolution
;    with the associated point-spread function.
;
; MODIFICATION HISTORY:
; 07/01/2011: Written by Lisa Dixon, New York University.
;   Based on RAYLEIGH_SOMMERFELD, whose modification history
;   follows:
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
; 01/19/2011: LD: Corrected deconvolution for arbitrary z ranges.
; 01/19/2011: DGG. Documentation fixes and formatting.
;
; Copyright (c) 2006-2011 Lisa Dixon, Sanghyuk Lee and David G. Grier
;
;-
function rs_decon, a, _z, $
                   lambda = lambda, $ ; wavelength of light
                   mpp = mpp, $       ; micrometers per pixel
                   nozphase = nozphase, $
                   hanning = hanning 

COMPILE_OPT IDL2

; hologram dimensions
sz = size(a)
ndim = sz[0]
if ndim gt 2 then message, "requires two-dimensional phase mask"
nx = double(sz[1])
if ndim eq 1 then $
   ny = 1. $
else $     
   ny = double(sz[2])

; volumetric slices
z = double(_z)
nz = n_elements(z)              ; number of z planes
minz = min(z, max = maxz)

; parameters
if n_elements(lambda) ne 1 then $
   lambda = 0.632D              ; HeNe laser
if n_elements(mpp) ne 1 then $
   mpp = 0.135D                 ; Nikon rig

ik = dcomplex(0., 2.D*!dpi * mpp / lambda) ; wavenumber in radians/pixel

; phase factor for Rayleigh-Sommerfeld propagator in Fourier space
; Refs. [2] and [3]
qx = dindgen(nx) - nx/2.
qx *= lambda / (nx * mpp)
qsq = qx^2

if ndim eq 2 then begin
   qy = dindgen(ny) - ny/2.D
   qy *= lambda / (ny * mpp)
   qsq = qsq # replicate(1.D, ny) + replicate(1.D, nx) # qy^2
endif

qfactor = sqrt(dcomplex(1.D - qsq))

if ~keyword_set(nozphase) then qfactor -= 1.D

if keyword_set(hanning) then qfactor *= hanning(nx, ny)
     
E = fft(a - mean(a), -1, /center) ; Fourier transform of input field

res = dcomplexarr(nx, ny, nz)
pt = dcomplexarr(nx, ny, nz)

for j = 0, nz-1 do begin
   if z[j] gt 0 then begin
      Hqz = exp(-ik * z[j] * conj(qfactor))  
      psz = exp(-ik * (z[j] - minz) * conj(qfactor))
   endif else begin
      Hqz = exp(-ik * z[j] * qfactor)
      psz = exp(-ik * (z[j] - maxz) * qfactor)
   endelse

;; psz = exp(-ik * float(z[j]-min(z)) * conj(qfactor))

; Convolve field with Rayleigh-Sommerfeld propagator
   thisE = E * Hqz                            ; Convolve with propagator 
   thisE = fft(thisE, 1, /center, /overwrite) ; Transform back to real space
   psz = fft(psz, 1, /center, /overwrite) 
   pt[*,*,j] = psz
   res[*,*,j] = thisE
endfor

;; deconvolve volume reconstruction with that of 
;; a simulated point scatter

outp = real_part(pt*conj(pt))
outr = real_part(res*conj(res))

ft_psf = fft(outp, -1, /center)
ft_hol = fft(outr, -1, /center)

out = fft(ft_hol/(ft_psf + 1e-2), +1, /center)

return, out

end
