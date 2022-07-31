using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ShapML
using ProgressMeter
using LaTeXStrings



# set the plotting theme
add_mints_theme()
theme(:mints)


p = Plots.showtheme(:mints)
savefig("theme.svg")

include("./config.jl")
include("./utils.jl")





datapath = "/media/john/HSDATA/processed/11-23"
outpath = "/media/john/HSDATA/analysis"

isdir(datapath)
isdir(outpath)


model = @load XGBoostRegressor pkg=XGBoost

mdl = model()
target = :CDOM
(y, X), (ytest, Xtest) = makeDatasets(datapath, target, 0.85)

hpo_model("XGBoostRegressor",
          model,
          target,
          targetsDict[target][2],
          targetsDict[target][1],
          X,
          y,
          Xtest,
          ytest,
          outpath;
          )





@showprogress for (target, info) ∈ targetsDict
    (y, X), (ytest, Xtest) = makeDatasets(datapath, target, 0.85)

    hpo_model("XGBoostRegressor",
              model,
              target,
              info[2],
              info[1],
              X,
              y,
              Xtest,
              ytest,
              outpath;
              )

    GC.gc()
end

