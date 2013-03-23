;+
; NAME:
;     rayleighsommerfeld
;
; PURPOSE:
;     Computes Rayleigh-Sommerfeld back-propagation of
;     a normalized hologram of the type measured by
;     digital video microscopy
;
; CATEGORY:
;     Holographic microscopy
;
; CALLING SEQUENCE:
;     b = rayleighsommerfeld(a, z, lambda, mpp)
;
; INPUTS:
;     a: hologram recorded as image data normalized
;         by a background image. 
;     z: displacement from the focal plane [pixels]
;         If z is an array of displacements, the field
;         is computed at each plane.
;     lambda: Wavelength of light in medium [micrometers]
;     mpp: Magnification [micrometer/pixel]
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
; IDL> lambda = 0.500
; IDL> mpp = 0.101
; IDL> for z = 1., 200 do begin $
; IDL> phiz = atan(rayleighsommerfeld(a,z,lambda,mpp),/phase) & $
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
; 11/07/2009: DGG  Code and documentation clean-up.  Properly
;   handle places where qsq > 1.  Implemented NOZPHASE.
; 11/09/2009: DGG Initial implementation of simultaneous projection
;   to multiple z planes for more efficient volumetric
;   reconstructions.  Eliminate matrix reorganizations.
; 06/10/2010: DGG Documentation fixes.  Added COMPILE_OPT.
; 07/27/2010: DGG Added HANNING.
; 10/20/2010: DGG Subtract 1 from (normalized) hologram
;   rather than expecting user to do subtraction.
; 06/24/2012 DGG Overhauled computation of Hqz.
; 03/17/2013 DGG More efficient array manipulations.
;   Require lamba and mpp as input parameters.  Check inputs.
;   Suppress floating point underflow errors.
; 03/22/2013 DGG rebin(/sample) is more efficient.
;
; Copyright (c) 2006-2013 Sanghyuk Lee and David G. Grier
;-
function rayleighsommerfeld, a, z, lambda, mpp, $
                             nozphase = nozphase, $
                             hanning = hanning 

COMPILE_OPT IDL2

umsg = 'result = rayleighsommerfeld(hologram, z, lambda, mpp)'

if n_params() ne 4 then begin
   message, umsg, /inf
   return, -1
endif

; hologram dimensions
sz = size(a)
ndim = sz[0]
if ~isa(a, /number, /array) || (ndim ne 2) then begin
   message, umsg, /inf
   message, 'HOLOGRAM must be a two dimensional numeric array', /inf
   return, -1
endif
nx = float(sz[1])
if ndim eq 1 then $
  ny = 1. $
else $
  ny = float(sz[2])

; volumetric slices
if ~isa(z, /number) then begin
   message, umsg, /inf
   message, 'Z should specify the height of the image plane in pixels', /inf
   return, -1
endif
nz = n_elements(z)              ; number of z planes

; parameters
if ~isa(lambda, /scalar, /number) then begin
   message, umsg, /inf
   message, 'LAMBDA should be the wavelength of light in the medium in micrometers', /inf
   return, -1
endif

if ~isa(mpp, /scalar, /number) then begin
   message, umsg, /inf
   message, 'MPP should be the magnification in micrometers per pixel', /inf
   return, -1
endif

ci = complex(0., 1.)
k = 2.*!pi*mpp/lambda           ; wavenumber in radians/pixel

; phase factor for Rayleigh-Sommerfeld propagator in Fourier space
; Refs. [2] and [3]
qx = findgen(nx)/nx - 0.5
qsq = ((lambda/mpp) * qx)^2

if ndim eq 2 then begin
   qy = findgen(1, ny)/ny - 0.5
   qsq = rebin(qsq, nx, ny, /sample) + rebin(((lambda/mpp) * qy)^2, nx, ny, /sample)
endif

qfactor = k * sqrt(complex(1. - qsq))

if ~keyword_set(nozphase) then $
   qfactor -= k

if keyword_set(hanning) then $
   qfactor *= hanning(nx, ny)

ikappa = ci * real_part(qfactor)
gamma = imaginary(qfactor) 

E = fft(complex(a - 1.), -1, /center) ; Fourier transform of input field
res = complexarr(nx, ny, nz, /nozero)

oexcept = !Except
!Except = 0
for j = 0, nz-1 do begin
   Hqz = exp(ikappa * z[j] - gamma * abs(z[j])) ; Rayleigh-Sommerfeld propagator
   thisE = E * Hqz                              ; convolve with propagator
   thisE = fft(thisE, 1, /center, /overwrite)   ; transform back to real space
   res[0, 0, j] = thisE                         ; save result
endfor
void = check_math()
!Except = oexcept

return, res

end
