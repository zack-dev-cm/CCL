# Direct number-counts reference for FKEM

`fkem_number_counts_reference.py` compares FKEM with direct spherical-Bessel
quadrature for density auto, density × (density + RSD), and (density + RSD) auto
spectra at multipoles 2, 5, and 10. It records a signed numerical error budget
for the common-grid integration discussed in
[issue #1307](https://github.com/LSSTDESC/CCL/issues/1307).

The script changes no production calculations and sets no spectrum acceptance
tolerance. Its purpose is to supply a reproducible reference case for choosing
an accuracy and integration convention.

## Definition and independence

The cosmology is flat LCDM with `Omega_c=.25`, `Omega_b=.05`, `h=.7`,
`sigma8=.8`, and `n_s=.96`; unspecified parameters retain CCL defaults.
The linear matter spectrum uses Eisenstein–Hu. The tracers use unit galaxy
bias, no magnification, and the normalized distribution

```text
n(z) = sin(pi*(z - 0.1)/1.2)^4 / 0.45,  0.1 <= z <= 1.3
     = 0,                              otherwise.
```

The independent angular projection is

```text
I_density(k) = integral dz n(z) D(z) j_ell(k chi(z))
I_RSD(k)     = -integral dz n(z) D(z) f(z) j_ell''(k chi(z))
C_ell       = 2/pi integral dlogk k^3 P(k, a=1) I_left(k) I_right(k).
```

The second derivative is with respect to the entire dimensionless Bessel
argument. Distances, growth, growth rate and the linear matter spectrum are
shared CCL inputs; the angular projection is independent of FFTLog and its
common grids. This checks the projection, not the background cosmology or the
Eisenstein–Hu implementation. Distances are in Mpc, wavenumbers in inverse Mpc,
power in cubic Mpc, and the resulting angular spectra are dimensionless.

Gauss–Legendre integration in redshift and composite Simpson integration in
logarithmic wavenumber produce the reference. A separate adaptive
Gauss–Kronrod integration checks the outer integral. The radial order,
wavenumber resolution and both wavenumber endpoints vary separately. These
are empirical convergence and domain-extension checks; they are not rigorous
bounds on the uncomputed infinite tail.

## Run

With NumPy, SciPy and a working CCL installation:

```bash
python benchmarks/data/codes/fkem_number_counts_reference.py \
  --output number-counts-reference.json \
  --grids-output number-counts-grids.json
pytest benchmarks/data/codes/test_fkem_number_counts_reference.py
```

The optional second JSON contains every returned native and common wavenumber
grid and the corresponding radial sampling. The ordinary report contains
versions, FKEM source hash, reference convergence, 27 FKEM comparisons,
input-sampling controls and manual/automatic/default Limber branch results.
`--nchi 512` restricts the comparison to one radial resolution.

The comparisons use 4097 input redshift samples, `fkem_chi_min=1.0`, and
`l_limber=100`, ensuring that the selected multipoles use FKEM. Input grids
with 1025, 2049 and 4097 samples are checked separately. The script asserts
that linear and nonlinear powers are the same `Pk2D` object; their Limber
contributions therefore cancel, making rescaling the returned spectrum a
valid measure-only diagnostic in this specific setup. The standard automatic
and default Limber branches are exercised separately without altering them.

## Reading the error budget

The signed components telescope, in the reported order, from the actual FKEM
value to the independent finite-domain reference. They are not unique,
independent causal attributions: interactions can move between components if
the order or reference convention changes.

- **Measure:** the difference after rescaling by common/native log spacing.
- **Remaining projection and domain residual:** radial-transform and input
  interpolation differences, any FKEM contribution outside the finite
  reference domain, and numerical integration noise.
- **Common-grid interpolation:** independent projections sampled at native
  grids, linearly interpolated to the common grid, versus independent
  projections evaluated directly there.
- **Endpoints:** the unweighted endpoint sum versus the trapezoidal rule.
- **Common-grid quadrature:** that trapezoid versus direct integration over
  the grid's overlap with the finite reference domain.
- **Overlap truncation:** the overlap integral versus the full finite-domain
  reference.

The finite reference domain is `[1e-7, 0.2]` inverse Mpc. Direct grid diagnostic
projections are set to zero outside it; the actual FKEM evaluation is not.
The report includes absolute errors as well as signed relative errors. The
latter are descriptive for these positive, nonzero spectra and should not be
reused near zero crossings.

The seven reference tests check analytic redshift normalization, a separate
Bessel-order recurrence and integration by parts. Their floating-point
comparison thresholds test mathematical identities; they do not prescribe
FKEM's scientific accuracy.

## Recorded observations

The accompanying [numerical report](../fkem_number_counts_reference.json) uses
CCL 3.3.6 at source revision `66b05e1a`, NumPy 1.26.4 and SciPy 1.15.3.
Across the nine reference spectra, the largest relative changes from radial
order, wavenumber resolution, lower-domain extension and upper-domain
extension are respectively `3.26e-11`, `1.77e-13`, `2.04e-15` and `3.58e-12`.
The adaptive outer integration agrees within `3.62e-11` relative. These
numbers describe this finite-domain calculation and its shared background
inputs, not a universal accuracy guarantee.

At `fkem_Nchi=512`, signed fractional differences from that reference are:

| ell | density auto | density × (density + RSD) | (density + RSD) auto |
| ---: | ---: | ---: | ---: |
| 2 | +1.10426e-2 | +4.37287e-3 | +1.00623e-2 |
| 5 | -1.70419e-3 | -6.96508e-4 | -3.18157e-4 |
| 10 | -5.02700e-4 | -2.00421e-4 | -8.81052e-5 |

Rescaling only the integration measure decreases absolute error for 6 of the
18 RSD-containing comparisons and increases it for the other 12. The
remaining projection/domain residual dominates the ell=2 budgets at
`fkem_Nchi=512`; omitted overlap ranges dominate the ell=5 and ell=10 budgets
at that resolution. This supports examining
those terms together before judging a weight change by total error.

Changing the input redshift sampling from 2049 to 4097 points changes the
FKEM spectra by at most `9.13e-7` relative in this case, separately from the
reference integration convergence above. Manual and automatic branches give
the same spectra on the selected multipoles; their returned transition
metadata is 100 and 10 respectively. The default returns -1 and uses Limber
for all three multipoles, giving different spectra as expected from that
choice. No transition policy is changed.

The integration convention, finite-range treatment and acceptable total
error remain choices for scientific review. These measurements do not
establish a survey likelihood or parameter bias.
