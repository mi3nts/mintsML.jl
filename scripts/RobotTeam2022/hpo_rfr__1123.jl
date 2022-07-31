using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ShapML
using ProgressMeter



# set the plotting theme
add_mints_theme()
theme(:mints)

Plots.showtheme(:mints)


include("./config.jl")
include("./utils.jl")

# set default resource for parallelization
MLJ.default_resource(CPUThreads())

datapath = "/media/john/HSDATA/processed/11-23"
outpath = "/media/john/HSDATA/analysis"

isdir(datapath)
isdir(outpath)


# try out functions
RFR = @load RandomForestRegressor pkg=DecisionTree



@showprogress for (target, info) ∈ targetsDict
    (y, X), (ytest, Xtest) = makeDatasets(datapath, target, 0.85)
    explore_model("RandomForestRegressor",
                  RFR,
                  target,
                  info[2],
                  info[1],
                  X,
                  y,
                  Xtest,
                  ytest,
                  outpath,
                  )

    GC.gc()
end




# target = :CDOM
# (y, X), (ytest, Xtest) = makeDatasets(datapath, target, 0.85)

# size(X)
# size(Xtest)

# schema(X)

# # load a model and check its type requirements

# plotattr("margins")

# explore_model("RandomForestRegressor",
#               RFR,
#               :CDOM,
#               targetsDict[:CDOM][2],
#               targetsDict[:CDOM][1],
#               X,
#               y,
#               Xtest,
#               ytest,
#               outpath,
#               )
