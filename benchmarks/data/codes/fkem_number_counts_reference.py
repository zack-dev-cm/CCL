"""Direct spherical-Bessel quadrature for density and RSD number counts.

The angular projection is independent of FFTLog and of FKEM's common k grid.
Background distances, growth and the linear matter spectrum are shared CCL
inputs. The comparison records errors; it does not set an acceptance tolerance.
"""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pyccl as ccl
import scipy
from scipy.integrate import quad_vec, simpson, trapezoid
from scipy.special import roots_legendre, spherical_jn

ZLO, ZHI = 0.1, 1.3
ELLS = (2, 5, 10)
CASES = {
    "density_auto": (0, 0),
    "density_density_rsd_cross": (0, 1),
    "density_rsd_auto": (1, 1),
}


def redshift_distribution(z):
    """Unit-normalized, compact sin^4 distribution in redshift."""
    z = np.asarray(z)
    result = np.sin(np.pi * (z - ZLO) / (ZHI - ZLO))**4
    return np.where((z >= ZLO) & (z <= ZHI), result, 0) / (
        (ZHI - ZLO) * 3 / 8)


def bessel_second_derivative(ell, x):
    """Second derivative with respect to the whole dimensionless argument."""
    return ((ell * (ell + 1) / x**2 - 1) * spherical_jn(ell, x)
            - 2 / x * spherical_jn(ell, x, derivative=True))


class Reference:
    """Direct radial projection using Gauss-Legendre nodes in redshift."""

    def __init__(self, cosmo):
        self.cosmo = cosmo

    @lru_cache(maxsize=8)
    def radial_rule(self, order):
        nodes, weights = roots_legendre(order)
        z = ZLO + (nodes + 1) * (ZHI - ZLO) / 2
        weights = weights * (ZHI - ZLO) / 2
        a = 1 / (1 + z)
        chi = ccl.comoving_radial_distance(self.cosmo, a)
        weights *= redshift_distribution(z)
        weights *= ccl.growth_factor(self.cosmo, a)
        rate = ccl.growth_rate(self.cosmo, a)
        return chi, weights, rate

    def project(self, k, ell, radial_order=1024):
        """Return separate density and RSD radial integrals, including sign."""
        k = np.atleast_1d(k)
        chi, weights, rate = self.radial_rule(radial_order)
        output = np.empty((2, len(k)))
        for start in range(0, len(k), 64):
            stop = min(start + 64, len(k))
            x = k[start:stop, None] * chi[None, :]
            output[0, start:stop] = spherical_jn(ell, x) @ weights
            output[1, start:stop] = (
                -bessel_second_derivative(ell, x) @ (weights * rate))
        return output

    def spectra_integrand(self, k, components):
        density, rsd = components
        total = density + rsd
        products = np.array([density**2, density * total, total**2])
        return (2 / np.pi * k**3
                * ccl.linear_matter_power(self.cosmo, k, 1.) * products)

    def integrate(self, ell, radial_order=1024, k_samples=2049,
                  kmin=1e-7, kmax=0.2):
        k = np.geomspace(kmin, kmax, k_samples)
        integrand = self.spectra_integrand(
            k, self.project(k, ell, radial_order))
        return simpson(integrand, x=np.log(k), axis=1)


def make_tracers(cosmo, samples):
    z = np.linspace(ZLO, ZHI, samples)
    dndz = (z, redshift_distribution(z))
    bias = (z, np.ones_like(z))
    return [ccl.NumberCountsTracer(cosmo, dndz=dndz, bias=bias, has_rsd=x)
            for x in (False, True)]


@contextmanager
def observe_grids(module):
    """Collect returned grids while preserving each original return value."""
    original = module._get_k_common
    records = []

    def observe(ks1, ks2):
        result = original(ks1, ks2)
        records.append(([np.array(k) for k in ks1],
                        [np.array(k) for k in ks2], np.array(result)))
        return result

    module._get_k_common = observe
    try:
        yield records
    finally:
        module._get_k_common = original


def describe_grid(k):
    return {"min": float(k[0]), "max": float(k[-1]), "n": len(k),
            "dlogk": float(np.log(k[-1] / k[0]) / (len(k) - 1))}


def error_components(reference, ell, case_index, observed, value,
                     full_reference):
    """An ordered telescoping budget, not a unique causal decomposition.

    The physical projection is set to zero outside [1e-7, 0.2] Mpc^-1 for
    this diagnostic. Separate domain extensions quantify this choice.
    The residual includes radial transforms, input interpolation, FKEM
    contributions outside that domain, and numerical integration noise.
    """
    left_grids, right_grids, k = observed
    kmin, kmax = 1e-7, 0.2
    common_step = np.log(k[-1] / k[0]) / (len(k) - 1)
    native = left_grids[0]
    native_step = np.log(native[-1] / native[0]) / (len(native) - 1)
    scaled = value * common_step / native_step

    def project_bounded(grid):
        inside = (grid >= kmin) & (grid <= kmax)
        result = np.zeros((2, len(grid)))
        result[:, inside] = reference.project(grid[inside], ell)
        return result

    direct = reference.spectra_integrand(k, project_bounded(k))[case_index]
    left_index, right_index = list(CASES.values())[case_index]

    def regrid(grids, tracer_index):
        result = np.interp(k, grids[0], project_bounded(grids[0])[0])
        if tracer_index:
            result += np.interp(k, grids[1], project_bounded(grids[1])[1])
        return result

    regridded = (2 / np.pi * k**3
                 * ccl.linear_matter_power(reference.cosmo, k, 1.)
                 * regrid(left_grids, left_index)
                 * regrid(right_grids, right_index))
    common_sum = float(np.sum(direct) * common_step)
    regridded_sum = float(np.sum(regridded) * common_step)
    trap = float(trapezoid(direct, x=np.log(k)))
    overlap = float(reference.integrate(
        ell, kmin=max(kmin, k[0]), kmax=min(kmax, k[-1]))[case_index])
    parts = {
        "measure": value - scaled,
        "remaining_projection_and_domain_residual": scaled - regridded_sum,
        "common_grid_interpolation": regridded_sum - common_sum,
        "endpoint_sum_minus_trapezoid": common_sum - trap,
        "common_grid_quadrature": trap - overlap,
        "overlap_truncation": overlap - full_reference,
    }
    error = value - full_reference
    return {
        "native_grids": [[describe_grid(x) for x in g]
                         for g in (left_grids, right_grids)],
        "common_grid": describe_grid(k),
        "weight_ratio_minus_one": float(native_step / common_step - 1),
        "signed_absolute_error": error,
        "signed_relative_error": error / full_reference,
        "parts": parts,
        "budget_closure": sum(parts.values()) - error,
        "measure_only_rescaled_value": scaled,
        "measure_only_rescaled_signed_error": scaled - full_reference,
        "measure_only_improves_absolute_error": bool(
            abs(scaled - full_reference) < abs(error)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grids-output", type=Path)
    parser.add_argument("--nchi", nargs="+", type=int,
                        default=[128, 256, 512])
    args = parser.parse_args()
    cosmo = ccl.Cosmology(
        Omega_c=.25, Omega_b=.05, h=.7, sigma8=.8, n_s=.96,
        transfer_function="eisenstein_hu", matter_power_spectrum="linear")
    cosmo.compute_linear_power()
    cosmo.compute_nonlin_power()
    if cosmo.get_linear_power() is not cosmo.get_nonlin_power():
        raise RuntimeError("Linear and nonlinear power must share one Pk2D")
    reference = Reference(cosmo)
    module = importlib.import_module("pyccl.nonlimber._nonlimber_FKEM")
    report = {
        "versions": {"pyccl": ccl.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "fkem_source_sha256": hashlib.sha256(
            Path(module.__file__).read_bytes()).hexdigest(),
        "definition": {
            "cosmology": {"Omega_c": .25, "Omega_b": .05, "h": .7,
                          "sigma8": .8, "n_s": .96},
            "linear_power": "eisenstein_hu",
            "linear_nonlinear_pk_identity": True,
            "redshift_support": [ZLO, ZHI],
            "redshift_distribution": "sin(pi*(z-.1)/1.2)^4 / .45",
            "density": "integral dz n(z) D(z) j_ell(k chi(z))",
            "rsd": "-integral dz n(z) D(z) f(z) j_ell''(k chi(z))",
            "spectrum": "2/pi integral dlogk k^3 P(k,1) I_left I_right",
            "units": {"chi": "Mpc", "k": "Mpc^-1", "P": "Mpc^3",
                      "C_ell": "dimensionless"},
            "shared_inputs": "CCL distances, growth, growth rate and power",
            "independent_part": "direct angular projection; no FFTLog",
            "reference_domain": [1e-7, .2],
            "acceptance_tolerance": None,
        },
        "reference_convergence": [], "comparisons": [],
        "input_sampling": [], "limber_branches": [], "adaptive_check": [],
    }
    configurations = [
        ("base", 1024, 2049, 1e-7, .2),
        ("radial_double", 2048, 2049, 1e-7, .2),
        ("k_double", 2048, 4097, 1e-7, .2),
        ("lower_domain", 2048, 4097, 1e-8, .2),
        ("upper_domain", 2048, 4097, 1e-7, .4),
    ]
    references = {}
    raw_grids = []
    for ell in ELLS:
        values = {}
        for label, nr, nk, lo, hi in configurations:
            values[label] = reference.integrate(ell, nr, nk, lo, hi)
            print(f"reference ell={ell} {label}", flush=True)
        for i, case in enumerate(CASES):
            central = values["k_double"][i]
            references[ell, case] = float(central)
            report["reference_convergence"].append({
                "ell": ell, "case": case,
                "values": {key: float(val[i]) for key, val in values.items()},
                "radial_delta": float(values["radial_double"][i]
                                      - values["base"][i]),
                "k_delta": float(central - values["radial_double"][i]),
                "lower_domain_delta": float(values["lower_domain"][i]
                                            - central),
                "upper_domain_delta": float(values["upper_domain"][i]
                                            - central),
            })

        def adaptive_integrand(logk):
            k = np.array([np.exp(logk)])
            return reference.spectra_integrand(
                k, reference.project(k, ell, 1024))[:, 0]

        adaptive, error, info = quad_vec(
            adaptive_integrand, np.log(1e-7), np.log(.2), epsabs=1e-18,
            epsrel=1e-10, limit=512, full_output=True)
        if not info.success:
            raise RuntimeError("Adaptive outer integration did not converge")
        report["adaptive_check"].append(dict(
            ell=ell, values=adaptive.tolist(),
            estimated_error=float(error), evaluations=info.neval,
            radial_order=1024))
    report["reference_configurations"] = [
        dict(label=x[0], radial_order=x[1], k_samples=x[2],
             kmin=x[3], kmax=x[4]) for x in configurations]
    tracers = make_tracers(cosmo, 4097)
    for nchi in args.nchi:
        for ell in ELLS:
            for index, (case, (left, right)) in enumerate(CASES.items()):
                with observe_grids(module) as grids:
                    val = float(ccl.angular_cl(
                        cosmo, tracers[left], tracers[right], ell,
                        l_limber=100, fkem_Nchi=nchi, fkem_chi_min=1.))
                if len(grids) != 1 or not np.isfinite(val):
                    raise RuntimeError("Expected one finite FKEM evaluation")
                if args.grids_output:
                    chi_max = max(x[-1] for x in tracers[0].get_kernel()[1])
                    raw_grids.append(dict(
                        ell=ell, case=case, nchi=nchi,
                        chi=np.geomspace(1., chi_max, nchi).tolist(),
                        left=[x.tolist() for x in grids[0][0]],
                        right=[x.tolist() for x in grids[0][1]],
                        common=grids[0][2].tolist()))
                budget = error_components(
                    reference, ell, index, grids[0], val,
                    references[ell, case])
                report["comparisons"].append(dict(
                    ell=ell, case=case, nchi=nchi, chi_min=1.,
                    value=val, reference=references[ell, case], **budget))
                print(f"comparison ell={ell} {case} Nchi={nchi}", flush=True)
    for samples in (1025, 2049, 4097):
        sampled = make_tracers(cosmo, samples)
        for case, (left, right) in CASES.items():
            vals = ccl.angular_cl(
                cosmo, sampled[left], sampled[right], np.array(ELLS),
                l_limber=100, fkem_Nchi=512, fkem_chi_min=1.)
            report["input_sampling"].append(dict(
                samples=samples, case=case,
                ell=list(ELLS), values=vals.tolist()))
    # Exercise the public defaults and automatic transition separately.
    for label, extra in [("manual", {"l_limber": 100}),
                         ("automatic", {"l_limber": "auto"}),
                         ("default", {})]:
        val, meta = ccl.angular_cl(
            cosmo, tracers[1], tracers[1], np.array(ELLS),
            fkem_Nchi=512, fkem_chi_min=1., return_meta=True, **extra)
        report["limber_branches"].append(dict(
            branch=label, ell=list(ELLS),
            values=val.tolist(),
            metadata={"l_limber": int(meta["l_limber"])}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.grids_output:
        args.grids_output.parent.mkdir(parents=True, exist_ok=True)
        args.grids_output.write_text(
            json.dumps(raw_grids, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
