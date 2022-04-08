mutable struct Bus
    bus_i::Int
    bustype::Int
    Pd::Float64
    Qd::Float64
    Gs::Float64
    Bs::Float64
    area::Int
    Vm::Float64
    Va::Float64
    baseKV::Float64
    zone::Int
    Vmax::Float64
    Vmin::Float64
end

mutable struct Line
    from::Int
    to::Int
    r::Float64
    x::Float64
    b::Float64
    rateA::Float64
    rateB::Float64
    rateC::Float64
    ratio::Float64 #TAP
    angle::Float64 #SHIFT
    status::Int
    angmin::Float64
    angmax::Float64
end
Line() = Line(0,0,0.,0.,0.,0.,0.,0.,0.,0.,0,0.,0.)

mutable struct Gener
    # .gen fields
    bus::Int
    Pg::Float64
    Qg::Float64
    Qmax::Float64
    Qmin::Float64
    Vg::Float64
    mBase::Float64
    status::Int
    Pmax::Float64
    Pmin::Float64
    Pc1::Float64
    Pc2::Float64
    Qc1min::Float64
    Qc1max::Float64
    Qc2min::Float64
    Qc2max::Float64
    ramp_agc::Float64
    # .gencost fields
    gentype::Int
    startup::Float64
    shutdown::Float64
    n::Int
    coeff::Array
end

mutable struct GenerVec{VI, VD}
    # .gen fields
    bus::VI
    Pg::VD
    Qg::VD
    Qmax::VD
    Qmin::VD
    Vg::VD
    mBase::VD
    status::VI
    Pmax::VD
    Pmin::VD
    Pc1::VD
    Pc2::VD
    Qc1min::VD
    Qc1max::VD
    Qc2min::VD
    Qc2max::VD
    ramp_agc::VD
    # .gencost fields
    gentype::VI
    startup::VD
    shutdown::VD
    n::VI
    #  coeff::Array
    coeff2::VD
    coeff1::VD
    coeff0::VD

    function GenerVec{VI, VD}(num_on) where {VI, VD}
        genvec = new{VI, VD}()
        genvec.bus = VI(undef, num_on)
        genvec.Pg = VD(undef, num_on)
        genvec.Qg = VD(undef, num_on)
        genvec.Qmax = VD(undef, num_on)
        genvec.Qmin = VD(undef, num_on)
        genvec.Vg = VD(undef, num_on)
        genvec.mBase = VD(undef, num_on)
        genvec.status = VI(undef, num_on)
        genvec.Pmax = VD(undef, num_on)
        genvec.Pmin = VD(undef, num_on)
        genvec.Pc1 = VD(undef, num_on)
        genvec.Pc2 = VD(undef, num_on)
        genvec.Qc1min = VD(undef, num_on)
        genvec.Qc1max = VD(undef, num_on)
        genvec.Qc2min = VD(undef, num_on)
        genvec.Qc2max = VD(undef, num_on)
        genvec.ramp_agc = VD(undef, num_on)
        genvec.gentype = VI(undef, num_on)
        genvec.startup = VD(undef, num_on)
        genvec.shutdown = VD(undef, num_on)
        genvec.n = VI(undef, num_on)
        genvec.coeff0 = VD(undef, num_on)
        genvec.coeff1 = VD(undef, num_on)
        genvec.coeff2 = VD(undef, num_on)
        return genvec
    end
end

struct OPFData
    buses::Array{Bus}
    lines::Array{Line}
    generators::Array{Gener}
    bus_ref::Int
    baseMVA::Float64
    BusIdx::Dict{Int,Int}    #map from bus ID to bus index
    FromLines::Array         #From lines for each bus (Array of Array)
    ToLines::Array           #To lines for each bus (Array of Array)
    BusGenerators::Array     #list of generators for each bus (Array of Array)
end

mutable struct Ybus{VD}
    YffR::VD
    YffI::VD
    YttR::VD
    YttI::VD
    YftR::VD
    YftI::VD
    YtfR::VD
    YtfI::VD
    YshR::Array{Float64}
    YshI::Array{Float64}

    Ybus{VD}(YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI) where {VD} = new(YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI)
end

function opf_loaddata(case_name, lineOff=Line())
    # import data as Dict(...)
    baseMVA = 100
    data = PowerModels.parse_file(case_name)
    PowerModels.standardize_cost_terms!(data, order=2)

    num_buses = length(data["bus"])

    buses = Array{Bus}(undef, num_buses)
    bus_ref=-1
    for (_, bus) in data["bus"]
        @assert bus["bus_i"] > 0  #don't support nonpositive bus ids
        i = bus["bus_i"]
        buses[i] = Bus(
            i,
            bus["bus_type"],
            0.0, 0.0, 0.0, 0.0,
            bus["area"],
            bus["vm"],
            bus["va"],
            bus["base_kv"],
            bus["zone"],
            bus["vmax"],
            bus["vmin"],
        )
        buses[i].Va *= pi/180
        if buses[i].bustype==3
            if bus_ref>0
                error("More than one reference bus present in the data")
            else
                bus_ref=i
            end
        end
    end

    for (_, load) in data["load"]
        load_bus = load["load_bus"]
        buses[load_bus].Pd = load["pd"] * baseMVA
        buses[load_bus].Qd = load["qd"] * baseMVA
    end

    for (_, shunt) in data["shunt"]
        shunt_bus = shunt["shunt_bus"]
        buses[shunt_bus].Gs = shunt["gs"]
        buses[shunt_bus].Bs = shunt["bs"]
    end

    num_lines = length(data["branch"])
    lines = Array{Line}(undef, num_lines)

    for (_, branch) in data["branch"]
        @assert branch["br_status"] == 1  # should be on since we discarded all other
        lit = branch["index"]
        lines[lit] = Line(
            branch["f_bus"],
            branch["t_bus"],
            branch["br_r"],
            branch["br_x"],
            branch["b_fr"] + branch["b_to"],
            branch["rate_a"] * baseMVA,
            branch["rate_b"] * baseMVA,
            branch["rate_c"] * baseMVA,
            branch["tap"],
            branch["shift"],
            branch["br_status"],
            branch["angmin"],
            branch["angmax"],
        )
    end

    num_gens = length(data["gen"])

    generators = Array{Gener}(undef, num_gens)
    i = 0
    for (_, gen) in data["gen"]
        if gen["gen_status"] == 1
            i += 1
            generators[i] = Gener(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, Array{Int}(undef, 0))

            generators[i].bus      = gen["gen_bus"]
            generators[i].Pg       = gen["pg"]
            generators[i].Qg       = gen["qg"]
            generators[i].Qmax     = gen["qmax"]
            generators[i].Qmin     = gen["qmin"]
            generators[i].Vg       = gen["vg"]
            generators[i].mBase    = gen["mbase"]
            generators[i].status   = gen["gen_status"]
            generators[i].Pmax     = gen["pmax"]
            generators[i].Pmin     = gen["pmin"]
            generators[i].Pc1      = gen["pc1"]
            generators[i].Pc2      = gen["pc2"]
            generators[i].Qc1min   = gen["qc1min"]
            generators[i].Qc1max   = gen["qc1max"]
            generators[i].Qc2min   = gen["qc2min"]
            generators[i].Qc2max   = gen["qc2max"]
            # generators[i].gentype  = costgen_arr[git,1]
            generators[i].startup  = gen["startup"]
            generators[i].shutdown = gen["shutdown"]
            generators[i].n        = gen["ncost"]
            generators[i].coeff    = gen["cost"] ./ [baseMVA^2, baseMVA, 1.0]
        end
    end

    # build a dictionary between buses ids and their indexes
    busIdx = mapBusIdToIdx(buses)

    # set up the FromLines and ToLines for each bus
    FromLines,ToLines = mapLinesToBuses(buses, lines, busIdx)

    # generators at each bus
    BusGeners = mapGenersToBuses(buses, generators, busIdx)

    #println(generators)
    #println(bus_ref)
    return OPFData(buses, lines, generators, bus_ref, baseMVA, busIdx, FromLines, ToLines, BusGeners)
end

function  computeAdmitances(lines, buses, baseMVA; VI=Array{Int}, VD=Array{Float64})
    nlines = length(lines)
    YffR=Array{Float64}(undef, nlines)
    YffI=Array{Float64}(undef, nlines)
    YttR=Array{Float64}(undef, nlines)
    YttI=Array{Float64}(undef, nlines)
    YftR=Array{Float64}(undef, nlines)
    YftI=Array{Float64}(undef, nlines)
    YtfR=Array{Float64}(undef, nlines)
    YtfI=Array{Float64}(undef, nlines)

    for i in 1:nlines
        @assert lines[i].status == 1
        Ys = 1/(lines[i].r + lines[i].x*im)
        #assign nonzero tap ratio
        tap = lines[i].ratio==0 ? 1.0 : lines[i].ratio

        #add phase shifters
        tap *= exp(lines[i].angle * pi/180 * im)

        Ytt = Ys + lines[i].b/2*im
        Yff = Ytt / (tap*conj(tap))
        Yft = -Ys / conj(tap)
        Ytf = -Ys / tap

        #split into real and imag parts
        YffR[i] = real(Yff); YffI[i] = imag(Yff)
        YttR[i] = real(Ytt); YttI[i] = imag(Ytt)
        YtfR[i] = real(Ytf); YtfI[i] = imag(Ytf)
        YftR[i] = real(Yft); YftI[i] = imag(Yft)
        #@printf("[%4d]  tap=%12.9f   %12.9f\n", i, real(tap), imag(tap));
    end

    nbuses = length(buses)
    YshR = Array{Float64}(undef, nbuses)
    YshI = Array{Float64}(undef, nbuses)
    for i in 1:nbuses
        YshR[i] = buses[i].Gs / baseMVA
        YshI[i] = buses[i].Bs / baseMVA
        #@printf("[%4d]   Ysh  %15.12f + %15.12f i \n", i, YshR[i], YshI[i])
    end

    @assert 0==length(findall(isnan.(YffR)))+length(findall(isinf.(YffR)))
    @assert 0==length(findall(isnan.(YffI)))+length(findall(isinf.(YffI)))
    @assert 0==length(findall(isnan.(YttR)))+length(findall(isinf.(YttR)))
    @assert 0==length(findall(isnan.(YttI)))+length(findall(isinf.(YttI)))
    @assert 0==length(findall(isnan.(YftR)))+length(findall(isinf.(YftR)))
    @assert 0==length(findall(isnan.(YftI)))+length(findall(isinf.(YftI)))
    @assert 0==length(findall(isnan.(YtfR)))+length(findall(isinf.(YtfR)))
    @assert 0==length(findall(isnan.(YtfI)))+length(findall(isinf.(YtfI)))
    @assert 0==length(findall(isnan.(YshR)))+length(findall(isinf.(YshR)))
    @assert 0==length(findall(isnan.(YshI)))+length(findall(isinf.(YshI)))

    if isa(VD, CuArray)
        return copyto!(VD(undef, nlines), 1, YffR, 1, nlines),
        copyto!(VD(undef, nlines), 1, YffI, 1, nlines),
        copyto!(VD(undef, nlines), 1, YttR, 1, nlines),
        copyto!(VD(undef, nlines), 1, YttI, 1, nlines),
        copyto!(VD(undef, nlines), 1, YftR, 1, nlines),
        copyto!(VD(undef, nlines), 1, YftI, 1, nlines),
        copyto!(VD(undef, nlines), 1, YtfR, 1, nlines),
        copyto!(VD(undef, nlines), 1, YtfI, 1, nlines),
        YshR, YshI
    else
        return YffR, YffI, YttR, YttI, YftR, YftI, YtfR, YtfI, YshR, YshI
    end
end


# Builds a map from lines to buses.
# For each line we store an array with zero or one element containing
# the  'From' and 'To'  bus number.
function mapLinesToBuses(buses, lines, busDict)
    nbus = length(buses)
    FromLines = [Int[] for b in 1:nbus]
    ToLines   = [Int[] for b in 1:nbus]
    for i in 1:length(lines)
        busID = busDict[lines[i].from]
        @assert 1<= busID <= nbus
        push!(FromLines[busID], i)

        busID = busDict[lines[i].to]
        @assert 1<= busID  <= nbus
        push!(ToLines[busID], i)
    end

    return FromLines,ToLines
end

# Builds a mapping between bus ids and bus indexes
#
# Returns a dictionary with bus ids as keys and bus indexes as values
function mapBusIdToIdx(buses)
    dict = Dict{Int,Int}()
    for b in 1:length(buses)
        @assert !haskey(dict,buses[b].bus_i)
        dict[buses[b].bus_i] = b
    end
    return dict
end


# Builds a map between buses and generators.
# For each bus we keep an array of corresponding generators number (as array).
#
# (Can be more than one generator per bus)
function mapGenersToBuses(buses, generators,busDict)
    gen2bus = [Int[] for b in 1:length(buses)]
    for g in 1:length(generators)
        busID = busDict[ generators[g].bus ]
        #@assert(0==length(gen2bus[busID])) #at most one generator per bus
        push!(gen2bus[busID], g)
    end
    return gen2bus
end

