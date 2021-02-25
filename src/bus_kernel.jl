function bus_kernel(
    baseMVA, nbus, pg_start, qg_start, pij_start, qij_start,
    pji_start, qji_start, wi_i_ij_start, wi_j_ji_start,
    FrStart, FrIdx, ToStart, ToIdx, GenStart, GenIdx,
    Pd, Qd, u, v, l, rho, YshR, YshI
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if I <= nbus
        common = 0
        inv_rhosum_pij_ji = 0
        inv_rhosum_qij_ji = 0
        rhosum_wi_ij_ji = 0

        @inbounds begin
            for k=FrStart[I]:FrStart[I+1]-1
                common += l[wi_i_ij_start+FrIdx[k]] + rho[wi_i_ij_start+FrIdx[k]]*u[wi_i_ij_start+FrIdx[k]]
                inv_rhosum_pij_ji += 1.0 / rho[pij_start+FrIdx[k]]
                inv_rhosum_qij_ji += 1.0 / rho[qij_start+FrIdx[k]]
                rhosum_wi_ij_ji += rho[wi_i_ij_start+FrIdx[k]]
            end
            for k=ToStart[I]:ToStart[I+1]-1
                common += l[wi_j_ji_start+ToIdx[k]] + rho[wi_j_ji_start+ToIdx[k]]*u[wi_j_ji_start+ToIdx[k]]
                inv_rhosum_pij_ji += 1.0 / rho[pji_start+ToIdx[k]]
                inv_rhosum_qij_ji += 1.0 / rho[qji_start+ToIdx[k]]
                rhosum_wi_ij_ji += rho[wi_j_ji_start+ToIdx[k]]
            end
        end

        common /= rhosum_wi_ij_ji

        rhs1 = 0
        rhs2 = 0
        inv_rhosum_pg = 0
        inv_rhosum_qg = 0

        @inbounds begin
            for g=GenStart[I]:GenStart[I+1]-1
                rhs1 += u[pg_start+GenIdx[g]] + (l[pg_start+GenIdx[g]]/rho[pg_start+GenIdx[g]])
                rhs2 += u[qg_start+GenIdx[g]] + (l[qg_start+GenIdx[g]]/rho[qg_start+GenIdx[g]])
                inv_rhosum_pg += 1.0 / rho[pg_start+GenIdx[g]]
                inv_rhosum_qg += 1.0 / rho[qg_start+GenIdx[g]]
            end

            rhs1 -= (Pd[I] / baseMVA)
            rhs2 -= (Qd[I] / baseMVA)

            for k=FrStart[I]:FrStart[I+1]-1
                rhs1 -= u[pij_start+FrIdx[k]] + (l[pij_start+FrIdx[k]]/rho[pij_start+FrIdx[k]])
                rhs2 -= u[qij_start+FrIdx[k]] + (l[qij_start+FrIdx[k]]/rho[qij_start+FrIdx[k]])
            end

            for k=ToStart[I]:ToStart[I+1]-1
                rhs1 -= u[pji_start+ToIdx[k]] + (l[pji_start+ToIdx[k]]/rho[pji_start+ToIdx[k]])
                rhs2 -= u[qji_start+ToIdx[k]] + (l[qji_start+ToIdx[k]]/rho[qji_start+ToIdx[k]])
            end

            rhs1 -= YshR[I]*common
            rhs2 += YshI[I]*common

            A11 = (inv_rhosum_pg + inv_rhosum_pij_ji) + (YshR[I]^2 / rhosum_wi_ij_ji)
            A12 = -YshR[I]*(YshI[I] / rhosum_wi_ij_ji)
            A21 = A12
            A22 = (inv_rhosum_qg + inv_rhosum_qij_ji) + (YshI[I]^2 / rhosum_wi_ij_ji)
            mu2 = (rhs2 - (A21/A11)*rhs1) / (A22 - (A21/A11)*A12)
            mu1 = (rhs1 - A12*mu2) / A11
            #mu = A \ [rhs1 ; rhs2]
            wi = common + ( (YshR[I]*mu1 - YshI[I]*mu2) / rhosum_wi_ij_ji )

            for k=GenStart[I]:GenStart[I+1]-1
                g = GenIdx[k]
                v[pg_start+g] = u[pg_start+g] + (l[pg_start+g] - mu1) / rho[pg_start+g]
                v[qg_start+g] = u[qg_start+g] + (l[qg_start+g] - mu2) / rho[qg_start+g]
            end
            for j=FrStart[I]:FrStart[I+1]-1
                k = FrIdx[j]
                v[pij_start+k] = u[pij_start+k] + (l[pij_start+k] + mu1) / rho[pij_start+k]
                v[qij_start+k] = u[qij_start+k] + (l[qij_start+k] + mu2) / rho[qij_start+k]
                v[wi_i_ij_start+k] = wi
            end
            for j=ToStart[I]:ToStart[I+1]-1
                k = ToIdx[j]
                v[pji_start+k] = u[pji_start+k] + (l[pji_start+k] + mu1) / rho[pji_start+k]
                v[qji_start+k] = u[qji_start+k] + (l[qji_start+k] + mu2) / rho[qji_start+k]
                v[wi_j_ji_start+k] = wi
            end
        end
    end
end

function bus_kernel_cpu(
    baseMVA, nbus, pg_start, qg_start, pij_start, qij_start,
    pji_start, qji_start, wi_i_ij_start, wi_j_ji_start,
    FrStart, FrIdx, ToStart, ToIdx, GenStart, GenIdx,
    Pd, Qd, u, v, l, rho, YshR, YshI
)
    Threads.@threads for I=1:nbus
        common = 0
        inv_rhosum_pij_ji = 0
        inv_rhosum_qij_ji = 0
        rhosum_wi_ij_ji = 0

        @inbounds begin
            if FrStart[I] < FrStart[I+1]
                common += sum(l[wi_i_ij_start+FrIdx[k]] + rho[wi_i_ij_start+FrIdx[k]]*u[wi_i_ij_start+FrIdx[k]] for k=FrStart[I]:FrStart[I+1]-1)
                inv_rhosum_pij_ji += sum(1.0 / rho[pij_start+FrIdx[k]] for k=FrStart[I]:FrStart[I+1]-1)
                inv_rhosum_qij_ji += sum(1.0 / rho[qij_start+FrIdx[k]] for k=FrStart[I]:FrStart[I+1]-1)
                rhosum_wi_ij_ji += sum(rho[wi_i_ij_start+FrIdx[k]] for k=FrStart[I]:FrStart[I+1]-1)
            end

            if ToStart[I] < ToStart[I+1]
                common += sum(l[wi_j_ji_start+ToIdx[k]] + rho[wi_j_ji_start+ToIdx[k]]*u[wi_j_ji_start+ToIdx[k]] for k=ToStart[I]:ToStart[I+1]-1)
                inv_rhosum_pij_ji += sum(1.0 / rho[pji_start+ToIdx[k]] for k=ToStart[I]:ToStart[I+1]-1)
                inv_rhosum_qij_ji += sum(1.0 / rho[qji_start+ToIdx[k]] for k=ToStart[I]:ToStart[I+1]-1)
                rhosum_wi_ij_ji += sum(rho[wi_j_ji_start+ToIdx[k]] for k=ToStart[I]:ToStart[I+1]-1)
            end
        end

        common /= rhosum_wi_ij_ji

        rhs1 = 0
        rhs2 = 0
        inv_rhosum_pg = 0
        inv_rhosum_qg = 0

        @inbounds begin
            if GenStart[I] < GenStart[I+1]
                rhs1 += sum(u[pg_start+GenIdx[g]] + (l[pg_start+GenIdx[g]]/rho[pg_start+GenIdx[g]]) for g=GenStart[I]:GenStart[I+1]-1)
                rhs2 += sum(u[qg_start+GenIdx[g]] + (l[qg_start+GenIdx[g]]/rho[qg_start+GenIdx[g]]) for g=GenStart[I]:GenStart[I+1]-1)
                inv_rhosum_pg += sum(1.0 / rho[pg_start+GenIdx[g]] for g=GenStart[I]:GenStart[I+1]-1)
                inv_rhosum_qg += sum(1.0 / rho[qg_start+GenIdx[g]] for g=GenStart[I]:GenStart[I+1]-1)
            end

            rhs1 -= (Pd[I] / baseMVA)
            rhs2 -= (Qd[I] / baseMVA)

            if FrStart[I] < FrStart[I+1]
                rhs1 -= sum(u[pij_start+FrIdx[k]] + (l[pij_start+FrIdx[k]]/rho[pij_start+FrIdx[k]]) for k=FrStart[I]:FrStart[I+1]-1)
                rhs2 -= sum(u[qij_start+FrIdx[k]] + (l[qij_start+FrIdx[k]]/rho[qij_start+FrIdx[k]]) for k=FrStart[I]:FrStart[I+1]-1)
            end

            if ToStart[I] < ToStart[I+1]
                rhs1 -= sum(u[pji_start+ToIdx[k]] + (l[pji_start+ToIdx[k]]/rho[pji_start+ToIdx[k]]) for k=ToStart[I]:ToStart[I+1]-1)
                rhs2 -= sum(u[qji_start+ToIdx[k]] + (l[qji_start+ToIdx[k]]/rho[qji_start+ToIdx[k]]) for k=ToStart[I]:ToStart[I+1]-1)
            end

            rhs1 -= YshR[I]*common
            rhs2 += YshI[I]*common

            A11 = (inv_rhosum_pg + inv_rhosum_pij_ji) + (YshR[I]^2 / rhosum_wi_ij_ji)
            A12 = -YshR[I]*(YshI[I] / rhosum_wi_ij_ji)
            A21 = A12
            A22 = (inv_rhosum_qg + inv_rhosum_qij_ji) + (YshI[I]^2 / rhosum_wi_ij_ji)
            mu2 = (rhs2 - (A21/A11)*rhs1) / (A22 - (A21/A11)*A12)
            mu1 = (rhs1 - A12*mu2) / A11
            #mu = A \ [rhs1 ; rhs2]
            wi = common + ( (YshR[I]*mu1 - YshI[I]*mu2) / rhosum_wi_ij_ji )

            for k=GenStart[I]:GenStart[I+1]-1
                g = GenIdx[k]
                v[pg_start+g] = u[pg_start+g] + (l[pg_start+g] - mu1) / rho[pg_start+g]
                v[qg_start+g] = u[qg_start+g] + (l[qg_start+g] - mu2) / rho[qg_start+g]
            end
            for j=FrStart[I]:FrStart[I+1]-1
                k = FrIdx[j]
                v[pij_start+k] = u[pij_start+k] + (l[pij_start+k] + mu1) / rho[pij_start+k]
                v[qij_start+k] = u[qij_start+k] + (l[qij_start+k] + mu2) / rho[qij_start+k]
                v[wi_i_ij_start+k] = wi
            end
            for j=ToStart[I]:ToStart[I+1]-1
                k = ToIdx[j]
                v[pji_start+k] = u[pji_start+k] + (l[pji_start+k] + mu1) / rho[pji_start+k]
                v[qji_start+k] = u[qji_start+k] + (l[qji_start+k] + mu2) / rho[qji_start+k]
                v[wi_j_ji_start+k] = wi
            end
        end
    end
end