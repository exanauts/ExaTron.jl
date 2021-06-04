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

mutable struct OPFData{VI, VD}
  buses::Array{Bus}
  lines::Array{Line}
  generators::Array{Gener}
  genvec::GenerVec{VI, VD}
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

function opf_loaddata(case_name, lineOff=Line(); VI=Array{Int}, VD=Array{Float64})
  #
  # load buses
  #
  # bus_arr = readdlm("data/" * case_name * ".bus")
  bus_arr = readdlm(case_name * ".bus")
  num_buses = size(bus_arr,1)
  buses = Array{Bus}(undef, num_buses)
  bus_ref=-1
  for i in 1:num_buses
    @assert bus_arr[i,1]>0  #don't support nonpositive bus ids
    buses[i] = Bus(bus_arr[i,1:13]...)
    buses[i].Va *= pi/180
    if buses[i].bustype==3
      if bus_ref>0
        error("More than one reference bus present in the data")
      else
         bus_ref=i
      end
    end
    #println("bus ", i, " ", buses[i].Vmin, "      ", buses[i].Vmax)
  end

  #
  # load branches/lines
  #
  # branch_arr = readdlm("data/" * case_name * ".branch")
  branch_arr = readdlm(case_name * ".branch")
  num_lines = size(branch_arr,1)
  lines_on = findall((branch_arr[:,11].>0) .& ((branch_arr[:,1].!=lineOff.from) .| (branch_arr[:,2].!=lineOff.to)) )
  num_on   = length(lines_on)

  if lineOff.from>0 && lineOff.to>0
    println("opf_loaddata: was asked to remove line from,to=", lineOff.from, ",", lineOff.to)
    #println(lines_on, branch_arr[:,1].!=lineOff.from, branch_arr[:,2].!=lineOff.to)
  end
  if length(findall(branch_arr[:,11].==0))>0
    println("opf_loaddata: ", num_lines-length(findall(branch_arr[:,11].>0)), " lines are off and will be discarded (out of ", num_lines, ")")
  end



  lines = Array{Line}(undef, num_on)

  lit=0
  for i in lines_on
    @assert branch_arr[i,11] == 1  #should be on since we discarded all other
    lit += 1
    lines[lit] = Line(branch_arr[i, 1:13]...)
	#=
    if (lines[lit].angmin != 0 || lines[lit].angmax != 0) && (lines[lit].angmin>-360 || lines[lit].angmax<360)
      println("Voltage bounds on line ", i, " with angmin ", lines[lit].angmin, " and angmax ", lines[lit].angmax)
      error("Bounds of voltage angles are still to be implemented.")
    end
	=#
  end
  @assert lit == num_on

  #
  # load generators
  #
  # gen_arr = readdlm("data/" * case_name * ".gen")
  gen_arr = readdlm(case_name * ".gen")
  # costgen_arr = readdlm("data/" * case_name * ".gencost")
  costgen_arr = readdlm(case_name * ".gencost")
  num_gens = size(gen_arr,1)

  baseMVA=100

  @assert num_gens == size(costgen_arr,1)

  gens_on=findall(x->x!=0, gen_arr[:,8]); num_on=length(gens_on)
  if num_gens-num_on>0
    println("loaddata: ", num_gens-num_on, " generators are off and will be discarded (out of ", num_gens, ")")
  end

  genvec = GenerVec{VI, VD}(num_on)
  copyto!(genvec.bus, 1, Int.(gen_arr[gens_on,1]), 1, num_on)
  copyto!(genvec.Pg, 1, gen_arr[gens_on,2] ./ baseMVA, 1, num_on)
  copyto!(genvec.Qg, 1, gen_arr[gens_on,3] ./ baseMVA, 1, num_on)
  copyto!(genvec.Qmax, 1, gen_arr[gens_on,4] ./ baseMVA, 1, num_on)
  copyto!(genvec.Qmin, 1, gen_arr[gens_on,5] ./ baseMVA, 1, num_on)
  copyto!(genvec.Vg, 1, gen_arr[gens_on,6], 1, num_on)
  copyto!(genvec.mBase, 1, gen_arr[gens_on,7], 1, num_on)
  copyto!(genvec.status, 1, Int.(gen_arr[gens_on,8]), 1, num_on)
  copyto!(genvec.Pmax, 1, gen_arr[gens_on,9] ./ baseMVA, 1, num_on)
  copyto!(genvec.Pmin, 1, gen_arr[gens_on,10] ./ baseMVA, 1, num_on)
  copyto!(genvec.Pc1, 1, gen_arr[gens_on,11], 1, num_on)
  copyto!(genvec.Pc2, 1, gen_arr[gens_on,12], 1, num_on)
  copyto!(genvec.Qc1min, 1, gen_arr[gens_on,13], 1, num_on)
  copyto!(genvec.Qc1max, 1, gen_arr[gens_on,14], 1, num_on)
  copyto!(genvec.Qc2min, 1, gen_arr[gens_on,15], 1, num_on)
  copyto!(genvec.Qc2max, 1, gen_arr[gens_on,16], 1, num_on)
  copyto!(genvec.gentype, 1, Int.(costgen_arr[gens_on,1]), 1, num_on)
  copyto!(genvec.startup, 1, costgen_arr[gens_on,2], 1, num_on)
  copyto!(genvec.shutdown, 1, costgen_arr[gens_on,3], 1, num_on)
  copyto!(genvec.n, 1, Int.(costgen_arr[gens_on,4]), 1, num_on)
  copyto!(genvec.coeff2, 1, costgen_arr[gens_on,5], 1, num_on)
  copyto!(genvec.coeff1, 1, costgen_arr[gens_on,6], 1, num_on)
  copyto!(genvec.coeff0, 1, costgen_arr[gens_on,7], 1, num_on)

  generators = Array{Gener}(undef, num_on)
  i=0
  for git in gens_on
    i += 1
    generators[i] = Gener(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, Array{Int}(undef, 0)) #gen_arr[i,1:end]...)

    generators[i].bus      = gen_arr[git,1]
    generators[i].Pg       = gen_arr[git,2] / baseMVA
    generators[i].Qg       = gen_arr[git,3] / baseMVA
    generators[i].Qmax     = gen_arr[git,4] / baseMVA
    generators[i].Qmin     = gen_arr[git,5] / baseMVA
    generators[i].Vg       = gen_arr[git,6]
    generators[i].mBase    = gen_arr[git,7]
    generators[i].status   = gen_arr[git,8]
    @assert generators[i].status==1
    generators[i].Pmax     = gen_arr[git,9]  / baseMVA
    generators[i].Pmin     = gen_arr[git,10] / baseMVA
    generators[i].Pc1      = gen_arr[git,11]
    generators[i].Pc2      = gen_arr[git,12]
    generators[i].Qc1min   = gen_arr[git,13]
    generators[i].Qc1max   = gen_arr[git,14]
    generators[i].Qc2min   = gen_arr[git,15]
    generators[i].Qc2max   = gen_arr[git,16]
    generators[i].gentype  = costgen_arr[git,1]
    generators[i].startup  = costgen_arr[git,2]
    generators[i].shutdown = costgen_arr[git,3]
    generators[i].n        = costgen_arr[git,4]
    @assert(generators[i].n <= 3 && generators[i].n >= 2)
    if generators[i].gentype == 1
      generators[i].coeff = costgen_arr[git,5:end]
      error("Piecewise linear costs remains to be implemented.")
    else
      if generators[i].gentype == 2
        generators[i].coeff = costgen_arr[git,5:end]
        #println(generators[i].coeff, " ", length(generators[i].coeff), " ", generators[i].coeff[2])
      else
        error("Invalid generator cost model in the data.")
      end
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
  return OPFData{VI, VD}(buses, lines, generators, genvec, bus_ref, baseMVA, busIdx, FromLines, ToLines, BusGeners)
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

