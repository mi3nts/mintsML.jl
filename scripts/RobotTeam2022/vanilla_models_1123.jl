using mintsML
using MLJ
using Plots, StatsPlots
using DataFrames, CSV
using ProgressMeter
using LaTeXStrings


# set the plotting theme
add_mints_theme()
theme(:mints)

# Plots.showtheme(:mints)


include("./config.jl")
include("./utils.jl")

# set default resource for parallelization
MLJ.default_resource(CPUThreads())

datapath = "/media/john/HSDATA/processed/11-23"
outpath = "/media/john/HSDATA/analysis"

isdir(datapath)
isdir(outpath)


ignore_models = ["ConstantRegressor",
                 "EvoTreeGaussian",
                 "TheilSenRegressor",
                 "LinearRegressor"]


# target = :CDOM
# # load an example dataset so we can find all compatible models.
# (y, X), (ytest, Xtest) = makeDatasets(datapath, target, 0.85)



# filter(model) = model.is_supervised &&
#     model.target_scitype >: scitype(y) &&
#     model.input_scitype >: scitype(X)

#     # model.input_scitype >: MLJ.Table(Continuous) &&
#     # model.target_scitype >: AbstractVector{<:Multiclass{3}} &&
#     # model.prediction_type == :deterministic


# mdls = models(filter)

# mdls[1]

# mdl_names = [mdl.name for mdl ∈ mdls]
# mdl_packages = [mdl.package_name for mdl ∈ mdls]
# mdl_hr_names = [mdl.human_name for mdl ∈ mdls]

# # # run this once to make sure we've got our environment setup
# # using Pkg
# # Pkg.add(unique(mdl_packages))
# # Pkg.add(unique(mlj_interfaces_base))


# mlj_interfaces = [load_path(mdl.name, pkg=mdl.package_name) for mdl ∈ mdls]
# mlj_interfaces_base = [split(interf, ".")[1] for interf ∈ mlj_interfaces]


# # now we can free up the space from the example dataset
# X = nothing
# y = nothing
# Xtest = nothing
# ytest = nothing
# GC.gc()


# # set up the dataframe for storing our results
# summary_df = DataFrame()
# summary_df.model_name = mdl_names
# summary_df.model_package = mdl_packages
# summary_df.model_name_long = mdl_hr_names
# summary_df.model_interface = mlj_interfaces

# # now let's go through each possible target and add blank columns for test and train r² score
# for (target, info) ∈ targetsDict
#     println(target)
#     summary_df[!, "$(target) train r²"] = zeros(size(summary_df, 1))
#     summary_df[!, "$(target) test r²"] = zeros(size(summary_df, 1))
# end


load_string = "model = @load NeuralNetworkRegressor pkg=MLJFlux"
eval(Meta.parse(load_string))
mdl = model()


# let's try it out for one of the models on one of the variables
summary_df = CSV.File(joinpath(outpath, "vanilla_comparison_1123.csv")) |> DataFrame

# now let us loop through each row, train the models, and update the dataframe
for row ∈ eachrow(summary_df)

    load_string = "model = @load $(row.model_name) pkg=$(row.model_package)"
    eval(Meta.parse(load_string))

    for (target, info) ∈ targetsDict
        if row["$(target) train r²"] == 0.0 && row["$(target) test r²"] == 0.0
            if !(row.model_name ∈ ignore_models)
                println("Working on model: $(row.model_name)\t target: $(target)")

                (y, X), (ytest, Xtest) = makeDatasets(datapath, target, 0.85)

                longname = targetsDict[target][2]
                units = targetsDict[target][1]
                model_name = row.model_name

                try
                    r²_train, r²_test = train_vanilla(model_name,
                                                      model,
                                                      target,
                                                      longname,
                                                      units,
                                                      X,
                                                      y,
                                                      Xtest,
                                                      ytest,
                                                      outpath
                                                      )

                    # update the DataFrame

                    row["$(target) train r²"] = r²_train
                    row["$(target) test r²"] = r²_test

                    # incrementally save the dataframe so we don't lose it.
                    CSV.write(joinpath(outpath, "vanilla_comparison_1123.csv"), summary_df)

                catch e
                    println(e)
                end
            end
        end
    end
end


# make heatmap of output
res = summary_df[!, Not([:model_name, :model_package, :model_name_long, :model_interface])]
data = Array(res)


x = [name for name ∈ names(summary_df) if !(name ∈ ["model_name", "model_package", "model_name_long", "model_interface"])]
x = [name[1:end-3] for name ∈ x]
y = summary_df.model_name

# set any zeros to NaN
data[data .<= 0.0]  .= NaN

pt = palette([:red, :limegreen], 10)

size(data)
size(x)
size(y)
heatmap(data,
        c=pt,
        clims=(0.0, 1.0),
        xticks=(1:1:size(x,1)+0.5, x),
        xrotation=90,
        yticks=(1:1:size(y,1)+0.5, y),
        size=(1200,900),
        yguidefontsize=5,
        colorbar_title=L"$R^2$ $\in$ $(0,1]",
        title="Vanilla Models for 11-23",
        grid = :none,
        minorgrid = :none,
        tick_direction = :none,
        minorticks = false,
        framestyle = :box,
        )

savefig(joinpath(outpath, "vanilla_comparison_1123.png"))
savefig(joinpath(outpath, "vanilla_comparison_1123.pdf"))
savefig(joinpath(outpath, "vanilla_comparison_1123.svg"))

