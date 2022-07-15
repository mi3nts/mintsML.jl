"""
    makeDatasets(datapath::String, target::Symbol, p::Float64)

Given a path to the TargetsAndFeatures.csv and a desired target variable. Return dataframes with a stratified split with percentage `p` as the training set.
"""
function makeDatasets(datapath::String, target::Symbol, p::Float64)
    # load data
    df = CSV.File(joinpath(datapath, "TargetsAndFeatures.csv")) |> DataFrame
    dropmissing!(df)

    # group by predye_postdye and only train on pre-dye
    gdf = groupby(df, :predye_postdye)
    data = gdf[(predye_postdye="Pre-Dye",)]

    # get rid of any missing

    ignored_for_input = [targets_vars..., ignorecols..., :MSR_705, :rad_MSR_705]


    # split into dev set and holdout
    df, df_test = partition(data,
                            p,  # save 10% for inter-model comparisons
                            stratify = data[!,target], # make sure we maintain target distribution
                            rng=42  # set the seed for reproducability
                            )

    # now we further split into targets and features
    y, X = unpack(df, ==(target), col -> !(col ∈ ignored_for_input))
    ytest, Xtest = unpack(df_test, ==(target), col -> !(col ∈ ignored_for_input))
    size(y)
    size(X)


    # if there's a third column in the targetsDict, set everything below it to 0.0
    if length(targetsDict[target]) == 3
        ymin = targetsDict[target][3]
        y[y .< ymin] .= 0.0
        ytest[ytest .< ymin] .= 0.0
    end

    return (y, X), (ytest, Xtest)
end



# assume default of 0.9 for p
function makeDatasets(datapath::String, target::Symbol)
    makeDatasets(datapath, target, 0.9)
end


