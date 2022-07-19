using LaTeXStrings
using Statistics
using StatsPlots: density, qqplot
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

    # define the marginal pdf
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






@userplot ScatterResult

@recipe function f(inputs::ScatterResult;)
    y = inputs.args[1]
    ŷ = inputs.args[2]
    ytest = inputs.args[3]
    ŷtest = inputs.args[4]



    ky = KernelDensity.kde(y)
    kŷ = KernelDensity.kde(ŷ)
    kytest = KernelDensity.kde(ytest)
    kŷtest = KernelDensity.kde(ŷtest)


    # compute r² scores for later use
    r2_train = r²(ŷ, y)
    r2_test = r²(ŷtest, ytest)

    # compute min/max for setting plot lims
    minval = minimum(vcat(y, ŷ, ytest, ŷtest))
    maxval = maximum(vcat(y, ŷ, ytest, ŷtest))

    # add a fudge factor to bounds to make sure we don't cut off any points
    δ = 0.2*(maxval - minval)
    minval = minval - 0.1*δ
    maxval = maxval + 0.1*δ

    # set layout for 3 panels (ignoring top right)
    layout := @layout [topdensity{0.9w,0.1h}             _
                       histogram{0.9w,0.9h}   rightdensity{0.1w,0.9h}  ]

    legend := :topleft
    xlabel --> "Truth"
    ylabel --> "Prediction"
    xlims --> (minval, maxval)
    ylims --> (minval, maxval)

    # add first series 1:1 line
    @series begin
        seriestype := :path
        color := :black
        alpha := 0.5
        subplot := 2
        label := "1:1"
        [minval, maxval], [minval, maxval]
    end

    # add series for training predictions
    @series begin
        seriestype := :scatter
        subplot := 2
        msw --> 0
        ms --> 3
        alpha --> 0.75
        color --> mints_palette[1]
        label := L"training $R^2 =$ %$(round(r2_train, digits=4))"

        y, ŷ
    end


    # add series for testing predictions
    @series begin
        seriestype := :scatter
        subplot := 2
        msws --> 0
        ms --> 3
        alpha --> 0.75
        color --> mints_palette[2]
        label := L"testing  $R^2 =$ %$(round(r2_test, digits=4))"

        ytest, ŷtest
    end




    # define the pdf for the top
    ticks := nothing
    xguide := ""
    yguide := ""
    @series begin
        seriestype := :density
        subplot := 1
        xlims := (minval, maxval)
        ylims := (-0.1, 1.1*maximum(maximum.([ky.density, kytest.density])))
        legend := false
        xlabel := ""
        ylabel := ""
        color --> mints_palette[1]

        y
    end

    @series begin
        seriestype := :density
        subplot := 1
        xlims := (minval, maxval)
        ylims := (-0.1, 1.1*maximum(maximum.([ky.density, kytest.density])))
        legend := false
        xlabel := ""
        ylabel := ""
        color --> mints_palette[2]

        ytest
    end



    @series begin
        seriestype := :density
        subplot := 3
        orientation := :h
        xlims := (-0.1, 1.1*maximum(maximum.([kŷ.density, kŷtest.density])))
        ylims := (minval, maxval)
        legend := false
        xlabel := ""
        ylabel := ""
        color --> mints_palette[1]

        ŷ
    end

    @series begin
        seriestype := :density
        subplot := 3
        orientation := :h
        xlims := (-0.1, 1.1*maximum(maximum.([kŷ.density, kŷtest.density])))
        ylims := (minval, maxval)
        legend := false
        xlabel := ""
        ylabel := ""
        color --> mints_palette[2]

        ŷtest
    end




end



# since we're not really doing anything special just make this a normal function
"""
    qunatilequantile(y, ŷ, ytest, ŷtest; kwargs...)

Create a quantile-qunatile plot of training and testing data.
"""
function quantilequantile(y, ŷ, ytest, ŷtest; kw...)
    # compute min/max for setting plot lims
    minval = minimum(vcat(y, ŷ, ytest, ŷtest))
    maxval = maximum(vcat(y, ŷ, ytest, ŷtest))

    # add a fudge factor to bounds to make sure we don't cut off any points
    δ = 0.2*(maxval - minval)
    minval = minval - 0.1*δ
    maxval = maxval + 0.1*δ

    p = qqplot(y,ŷ;
               xlims=(minval,maxval),
               ylims=(minval,maxval),
               xlabel="Truth",
               ylabel="Prediction",
               msw=0,
               ms=3,
               alpha=0.75,
               label="Training",
               kw...,
               )

    qqplot!(p,
            ytest,ŷtest;
            color = mints_palette[2],
            msw = 0,
            ms = 3,
            alpha = 0.75,
            label = "Testing",
#            legend=:topleft,
            legend = :outertopright,
            kw...,
            )
    return p

end





@userplot RankImportances

@recipe function f(inputs::RankImportances;)
    """
        Assume you've been given a list of sorted features and their relative importances
    """
    feature_names = inputs.args[1]
    rel_importance = inputs.args[2]

    # first, we want to sort everyting

    idx = sortperm(rel_importance, rev=true)
    feature_names = feature_names[idx]
    rel_improtance = rel_importance[idx]

    # generate color palette and vector of colors for each bar
    p = palette([mints_palette[2], mints_palette[1]], 11)  # i.e. from 0.0:0.1:1.0
    colors = [p[Int(10*round(imp,digits=1))+1] for imp ∈ rel_importance]

    # generate the labels for each bar
    labels = (0.5:1:size(rel_importance,1)+1, feature_names)

    @series begin
        seriestype := :bar
        permute := (:y, :x)
        color := colors
        ylims --> (-0.01, 1.01)
        xflip --> true
        xticks --> labels
        legend --> false
        grid := :none
        minorgrid := :none
        tick_direction := :none
        minorticks := false
        framestyle := :box
        label := ""
        ylabel --> "relative mean |shap efect|"

        feature_names, rel_importance
    end
end



