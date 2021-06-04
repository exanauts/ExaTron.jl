using Plots

function draw_heatmap(filename,nx,ny;scale=1e3)
    # x: iteration
    # y: processor

    fin = open(filename, "r")
    lines = readlines(fin)
    z = zeros(nx,ny)

    for l in lines
        cols = split(l)
        y = parse(Int, cols[1])
        for x=1:6
            z[x,y] = scale * parse(Float64, cols[x+1])
        end
    end


    return z
end

z = draw_heatmap(ARGS[1], 6, 41126)
savefig(heatmap(z[:,35000:36000], xlabel="ADMM Iteration", ylabel="GPU", colorbar_title="Time (ms)"), splitext(ARGS[1])[1]*".pdf")
