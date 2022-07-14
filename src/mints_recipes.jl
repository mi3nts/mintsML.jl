using Statistics
using StatsPlots: density
using DataFrames
import KernelDensity


@userplot HistPDF

@recipe function f(mpdf::HistPDF;)
    x = mpdf.args[1]

    k = KernelDensity.kde(x)

    # compute the 1% and 99% quantiles to set the xlims
    # xmin = quantile(x, 0.01)
    # xmax = quantile(x, 0.99)



    # layout = @layout [topdensity
    #                   histogram{1.0w, 0.85h}]
    link := :x
    layout := @layout [topdensity             _
                       histogram{1.0w,0.9h}   _  ]

    # define the histogram
    @series begin
        seriestype := :histogram
        subplot := 2
        lw --> 0.25
        alpha --> 0.75
        nbins --> 100
        alpha --> 0.75
#        xlims --> (xmin, xmax)
        ylabel := "Count"
        legend --> :topright
        x
    end

    # define the marginal histogram
    ticks := nothing
    xguide := ""
    yguide := ""

    @series begin
        seriestype := :density
        subplot := 1
#        xlims --> (xmin, xmax)
        ylims --> (-0.1, 1.1*maximum(k.density))
        legend := false
        xlabel := ""
        ylabel := ""

        x
    end
end







@userplot CorrMatrix

@recipe function f(inputs::CorrMatrix)
    df = inputs.args[end]
    var_names = names(df)


    corr = cor(Matrix(df))

    @series begin
        seriestype := :heatmap
        xticks := (1:ncol(df), var_names)
        yticks := (1:ncol(df), var_names)
        xrotation := 90
        guidefontsize := 10
        yflip := true
        grid := :none
        minorgrid := :none
        tick_direction := :none
        minorticks := false
        framestyle := :box
        corr
    end
end




