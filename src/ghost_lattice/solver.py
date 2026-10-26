"""Run a single explicitly requested case with independent CPU force acceptance."""

from contextlib import nullcontext
import copy
import json
import time
import types
from pathlib import Path
import numpy as np
from .cases import Case, get_case
from .models import build


def _json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _cuda(model):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build or explicitly select --backend cpu."
        )
    torch.set_num_threads(1)
    if model.X.shape[1] == 3:
        if model.plast == "cp":
            from .numerics.cuda_bcc_slip import attach
        else:
            from .numerics.cuda_bcc_j2 import attach
        attach(model)
    elif model.plast == "cp":
        from .numerics.cuda_slip import GPUSlip

        reference = model.constitutive_cp
        local = GPUSlip(model)
        evidence = []
        calls = 0

        def checked(dL):
            nonlocal calls
            calls += 1
            answer = local.update(dL)
            if calls <= 3 or calls % 100 == 0:
                original = reference(dL)
                differences = [
                    float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-14))
                    for a, b in zip(original[:3], answer[:3])
                ]
                if max(differences) > 2e-7:
                    raise RuntimeError(
                        f"Independent local CP CPU/GPU comparison failed: {differences}"
                    )
                evidence.append({"call": calls, "differences": differences})
            return answer

        model.constitutive_cp = checked
        model._public_gpu_checks = evidence
    else:
        from .numerics.cuda_rotated_j2 import attach
        from .numerics.cuda_anderson import step as anderson

        attach(model)
        if model.kind == "j04":
            solver = model._gpu_solver
            newton = solver.step

            def robust(self, E, load, tol=1e-9, **kwargs):
                try:
                    return newton(E, load, tol, **kwargs)
                except RuntimeError:
                    return anderson(self, E, load, tol, **kwargs)

            solver.step = types.MethodType(robust, solver)


def run(
    name,
    output,
    method="proposed",
    backend="cpu",
    data_root=None,
    seed=None,
    assembly=None,
):
    case = name if isinstance(name, Case) else get_case(name, seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Use an empty output directory to preserve existing results."
        )
    output.mkdir(parents=True, exist_ok=True)
    from .numerics.mls_safeguard import MLSGuard

    guard = MLSGuard()
    context = (
        guard.installed()
        if case.geometry not in ("equiaxed", "columnar")
        else nullcontext()
    )
    started = time.perf_counter()
    try:
        with context:
            built = build(case, method, data_root, assembly)
    except Exception as error:
        _json(
            output / "result.json",
            {"status": "failed", "stage": "model_construction", "error": str(error)},
        )
        raise
    _json(
        output / "configuration.json",
        {
            **built.diagnostics,
            "backend": backend,
            "assembly": assembly,
            "MLS_safeguard": guard.report(),
        },
    )
    if case.law == "elastic":
        info = built.model.solve(built.load_tensor)
        if info["relative_force_residual"] > 1.05e-9:
            raise RuntimeError("Independent elastic force equilibrium failed")
        obj = built.model
        points = obj.m.X * case.spacing
        window = np.all(
            (points > case.window_margin) & (points < case.size - case.window_margin),
            axis=1,
        )
        np.savez_compressed(
            output / "load_1.npz",
            X=points,
            win=window,
            u=obj.u * case.spacing,
            grain=obj.grain,
            strain=obj.strain,
            stress=obj.stress,
        )
        _json(
            output / "result.json",
            {"status": "completed", "steps": 1, "equilibrium": info},
        )
        return
    model = built.model
    if model.plast == "cp" and model.X.shape[1] == 2:
        from .numerics.implicit_slip import attach

        attach(model)
    if backend == "cuda":
        _cuda(model)
    elif backend != "cpu":
        raise ValueError("backend must be cpu or cuda")
    X = model.X * built.length_per_lattice_unit
    dim = X.shape[1]
    axis = slice(0, 2) if case.geometry == "columnar" else slice(0, dim)
    window = np.all(
        (X[:, axis] > case.window_margin)
        & (X[:, axis] < case.size - case.window_margin),
        axis=1,
    )
    if dim == 2:
        from .numerics.strain_reconstruction import strain_operator_any_order

        exporter = strain_operator_any_order(model)
    history, snapshots = [], []
    loads = case.loads()
    stage = "equilibrium"
    try:
        for step, load in enumerate(loads, 1):
            stage = "equilibrium"
            previous = {
                key: copy.copy(getattr(model, key))
                for key in ("u", "alpha", "epsp", "dLp", "lam")
            }
            tick = time.perf_counter()
            try:
                count = int(
                    model.step(
                        built.load_tensor,
                        float(load) * built.strain_scale,
                        tol=1e-9,
                        maxit=600,
                        mAA=8,
                    )
                )
                force, _ = model.forces(model.B @ model.u, model.dLp, model.epsp)
                residual = float(
                    np.linalg.norm(force[model.free])
                    / max(np.linalg.norm(force[model.fixed]), 1e-30)
                )
                if count >= 599 or not np.isfinite(residual) or residual > 1.05e-9:
                    raise RuntimeError(
                        f"Equilibrium acceptance failed at step {step}: iterations={count}, CPU residual={residual}"
                    )
            except Exception:
                for key, value in previous.items():
                    setattr(model, key, value)
                raise
            history.append(
                {
                    "step": step,
                    "load": float(load),
                    "iterations": count,
                    "independent_CPU_force_residual": residual,
                    "seconds": time.perf_counter() - tick,
                }
            )
            stage = "resistance_update"
            built.update_resistance()
            # The zero checkpoint in reversal is on the return leg.
            save = any(abs(load - target) < 1e-10 for target in case.checkpoints)
            if case.reverse and abs(load) < 1e-10:
                save = step > round(case.maximum_load / case.increment)
            if save:
                stage = "field_export"
                field = dict(
                    X=X,
                    u=model.u * built.length_per_lattice_unit,
                    grain=model.grain,
                    win=window,
                    alpha=model.alpha,
                    epsp=model.epsp,
                    dgb=model.dgb * built.length_per_lattice_unit,
                )
                if dim == 2:
                    tensor = (exporter @ (model.B @ model.u)).reshape(model.N, 3)
                    field["strain"] = tensor * [1.0, 1.0, 2.0]
                    field["ges"] = np.hypot(
                        (tensor[:, 0] - tensor[:, 1]) / 2, tensor[:, 2]
                    )
                file = f"step_{step:04d}.npz"
                np.savez_compressed(output / file, **field)
                snapshots.append({"step": step, "load": float(load), "path": file})
            _json(
                output / "progress.json",
                {
                    "status": "running",
                    "accepted_steps": step,
                    "total_steps": len(loads),
                    "load": float(load),
                    "CPU_force_residual": residual,
                },
            )
    except Exception as error:
        _json(
            output / "result.json",
            {
                "status": "failed",
                "stage": stage,
                "accepted_steps": len(history),
                "total_steps": len(loads),
                "error": str(error),
                "history": history,
                "snapshots": snapshots,
            },
        )
        raise
    _json(
        output / "result.json",
        {
            "status": "completed",
            "N": model.N,
            "steps": len(history),
            "history": history,
            "snapshots": snapshots,
            "seconds_including_setup": time.perf_counter() - started,
            "local_CPU_GPU_checks": getattr(model, "_public_gpu_checks", []),
            "scope": "A fresh run generated from the selected public Case definition.",
        },
    )
