
import ExaPF.PowerSystem: ParsePSSE, ParseMAT

function run_pf_ref(datafile::String)
    polar = ExaPF.PolarForm(datafile)
    buffer = get(polar, ExaPF.PhysicalState())
    ExaPF.init_buffer!(polar, buffer)

    jx = ExaPF.AutoDiff.Jacobian(polar, ExaPF.power_balance, State())
    ls = ExaPF.LinearSolvers.DirectSolver(jx.J)

    ExaPF.powerflow(polar, jx, buffer, NewtonRaphson(verbose=1); linear_solver=ls)
    return buffer
end

function run_pf(datafile::String; options...)
    env = ExaTron.admm_rect_gpu_two_level(datafile; options...)
    return (
        env=env,
        vm=ExaTron.voltage_magnitude(env),
        va=ExaTron.voltage_angle(env),
        pg=ExaTron.active_power_generation(env),
        qg=ExaTron.reactive_power_generation(env),
    )
end

# Import directly data from matpower file
function load_matpower_file(datafile::String)
    data_mat = ParseMAT.parse_mat(datafile)
    data = ParseMAT.mat_to_exapf(data_mat)
    return ExaTron.RawData(
        data["baseMVA"][1],
        data["bus"],
        data["branch"],
        data["gen"],
        data["cost"],
    )
end

function run_pf_mat_acopf(datafile::String;
    outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
    use_gpu=false, use_polar=true, gpu_no=0, verbose=1, outer_eps=2e-4,
)
    raw = load_matpower_file(datafile)
    data = ExaTron.opf_loaddata(raw)
    env = ExaTron.AdmmEnv(
        data, use_gpu, rho_pq, rho_va; use_polar=use_polar, type=:opf_two_level, gpu_no=gpu_no, verbose=verbose,
    )
    ExaTron.admm_restart!(env, outer_iterlim=outer_iterlim, inner_iterlim=inner_iterlim, scale=scale)
    return (
        env=env,
        vm=ExaTron.voltage_magnitude(env),
        va=ExaTron.voltage_angle(env),
        pg=ExaTron.active_power_generation(env),
        qg=ExaTron.reactive_power_generation(env),
    )
end

function run_pf_mat_pf(datafile::String;
    outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0, scale=1e-4,
    use_gpu=false, use_polar=true, gpu_no=0, verbose=1
)
    raw = load_matpower_file(datafile)
    data = ExaTron.opf_loaddata(raw)
    env = ExaTron.AdmmEnv(data, use_gpu, rho_pq, rho_va; use_polar=use_polar, gpu_no=gpu_no, verbose=verbose, type=:power_flow)
    sol = env.solution
    ExaTron.admm_solve!(env, sol; outer_iterlim=outer_iterlim, inner_iterlim=inner_iterlim, scale=scale)
    return (
        env=env,
        vm=ExaTron.voltage_magnitude(env),
        va=ExaTron.voltage_angle(env),
        pg=ExaTron.active_power_generation(env),
        qg=ExaTron.reactive_power_generation(env),
    )
end

