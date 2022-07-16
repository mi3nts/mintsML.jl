using mintsML
using MLJ
using Plots
using DataFrames, CSV
using ShapML


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


# -------------- demo first for CDOM -------------------

target = :CDOM
(y, X), (ytest, Xtest) = makeDatasets(datapath, target, 0.85)

size(X)
size(Xtest)


histogram(y, alpha=0.7, label="train")
histogram!(ytest, alpha=0.7, label="test")


schema(X)

# load a model and check its type requirements
RFR = @load RandomForestRegressor pkg=DecisionTree



# instantiate model with default hyperparameters
forest = RFR()

# make sure model will work for our targets
scitype(y) <: target_scitype(forest)
scitype(X) <: input_scitype(forest)



# per this link: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# n_trees, and n_subfeatures are most important

mach = machine(forest, X, y)
fit!(mach)

fp = fitted_params(mach);
keys(fp)

ŷ = MLJ.predict(mach, X)
ŷtest = MLJ.predict(mach, Xtest)


scatterresult(y, ŷ, ytest, ŷtest; plot_title="RFR results for $(target)")


# # now make the quantile-quantile plot
# pqq = qqplot(target_train, pred_train,
#             color=:green,
#             alpha=0.5,
#             label="Training samples",
#             legend=true,
#             #xtickfontsize=18,
#             #ytickfontsize=18,
#             #xguidefontsize=20,
#             #yguidefontsize=20,
#             #titlefontsize=22,
#             #size=(900,900)
#             )
# qqplot!(target_test, pred_test,
#       color=:red,
#       alpha=0.5,
#       label="Testing samples",
#       legend=true,
#       )
# title!("Quantile-Quantile plot for $(target)")
# xlabel!("$(target) true")
# ylabel!("$(target) prediction")














# println("\tRanking feature importance")
# # add the SHAP value calculation here
# function predict_function(model, data)
#     data_pred = DataFrame(y_pred = MLJ.predict(model, data))
#     return data_pred
# end

# Nexplain = 300
# sample_size = 60
# explain = Xtrain[1:Nexplain, :]
# # Compute stochastic Shapley values.
# data_shap = ShapML.shap(explain = explain,
#                         model = mach,
#                         predict_function = predict_function,
#                         sample_size = sample_size,
#                         seed = 42
#                         )
# # group by feature name
# g_datashap = groupby(data_shap, :feature_name)
# # aggregate by mean abs shap value
# data_plot = combine(g_datashap, :shap_effect => (x->mean(abs.(x))) => :mean_effect)
# # sort by mean_effect
# idx = sortperm(data_plot.mean_effect)

# plot_features = data_plot.feature_name[idx]
# plot_vals = data_plot.mean_effect[idx]
# pbar = bar(plot_features[1:20], plot_vals[1:20] , yticks=(0.5:1:19.5, plot_features[1:20]), orientation = :horizontal, label="")
# xlabel!("Mean Absolute SHAP value")
# title!("Ranked Feature Importance for $(targetName)")




























# # since this is a tree model, extract feature importances from the report
# rpt = report(mach)
# keys(rpt)
# fi = rpt.evo_tree_classifier.feature_importances
# feature_importance_table = (feature=Symbol.(first.(fi)), importance=last.(fi)) |> DataFrames.DataFrame


# sort!(feature_importance_table, :importance, rev=true)

# fi_df = feature_importance_table[1:15, :]

# labels = String.(fi_df.feature)
# bar(fi_df.importance, yticks=(1:nrow(fi_df), labels), orientation=:horizontal, yflip=true, legend=false,label="")


# # for models not supporting feature_importances we should use shapely.jl for model agnostic feature importances. We should do that by default and then if they exist, also do the above ranking.

# ŷ = predict(mach_pipe, rows=validation)

# print(
#     "Measurements:\n",
#     "\tbrier loss: ", brier_loss(ŷ, y[validation]) |> mean, "\n",
#     "\tauc:        ", auc(ŷ, y[validation]), "\n",
#     "\taccuracy:   ", accuracy(mode.(ŷ), y[validation])   # we need mode to generate point predictions
# )

# # we could have also used `predict_mode` to generate the predictions
# # generate confusion matrix and ROC curve

# confmat(mode.(ŷ), y[validation])
# roc_curve = roc(ŷ, y[validation])

# plt = scatter(roc_curve, legend=false)
# plot!(plt, xlabel="false positive rate", ylabel="true positive rate")
# plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black)




# #  2. automated performance evaluation (the *more typical* workflow)

# # we will evaluate using a stratified cross validation instead of the holdout validation set
# e_pipe = evaluate(
#     pipe, X, y,
#     resampling = StratifiedCV(nfolds=10, rng=42),
#     measures=[brier_loss, auc, accuracy],
#     repeats=3,
#     acceleration=CPUThreads()
# )

# # use std on CV folds to estimate confidence interval for performance metrics
# # see: https://arxiv.org/abs/2104.00673 for why this may be a bad choice
# using Measurements

# function confidence_intervals(e)
#     factor = 2.0 # to get level of 95%
#     measure = e.measure
#     nfolds = length(e.per_fold[1])
#     measurement = [e.measurement[j] ± factor*std(e.per_fold[j])/sqrt(nfolds - 1)
#                    for j in eachindex(measure)]
#     table = (measure=measure, measurement=measurement)
#     return DataFrames.DataFrame(table)
# end

# confidence_intervals_basic_model = confidence_intervals(e_pipe)


# # ---------------- filtering out unimportant features ------------------------
# # let's use our fitted model to update our pipeline
# # drop those features with low feature importance to allow us to speed up optimization later
# unimportant_features = filter(:importance => <(0.005), feature_importance_table).feature
# pipe2 = ContinuousEncoder() |> FeatureSelector(features=unimportant_features, ignore=true) |> booster

# # for iterative models (i.e. models that train through a set of steps), it is convenient to set
# # a list of controls that define when to stop training. We do this with a wrapper that makes our
# # pipe into an 'IteratedModel'

# # the list of controls are here: https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided

# controls = [
#     Step(1),              # to increment iteration parameter (`pipe.nrounds`)
#     NumberSinceBest(4),   # main stopping criterion
#     TimeLimit(2/3600),    # never train more than 2 sec
#     InvalidValue()        # stop if NaN or ±Inf encountered
# ]

# # now wrap our pipeline with these controls into an `IteratedModel`
# iterated_pipe = IteratedModel(model=pipe2,
#                               controls=controls,
#                               measure=brier_loss,
#                               resampling=Holdout(fraction_train=0.7)  # i.e. at each training step, evaluate against a holdout fraction of our supplied training data
#                               )

# mach_iterated_pipe = machine(iterated_pipe, X, y)  # we bind to all data not in test set
# fit!(mach_iterated_pipe)

# # summary of what's happening:
# # - controlled iteration step to tune hyperparameters which uses the holdout set to evaluate performance metric and stops when the control criteria are met
# # - a final training step that trains atomic model on *all* of the data using number of iterations determined in step 1.




# # Hyper-parameter optimization (model tuning)
# # we choose to optimize both `max_depth` and `η`
# # just like iteration control, hyperparameter optimization is done with a model wrapper: `TunedModel`
# show(iterated_pipe, 2)  # show 2 levels of parameters since it's nested

# # define the learnable hyperparameters
# p1 = :(model.evo_tree_classifier.η)
# p2 = :(model.evo_tree_classifier.max_depth)

# # define allowable ranges
# r1 = range(iterated_pipe, p1, lower=-2, upper=-0.5, scale=x->10^x)  # make it logarithmic
# r2 = range(iterated_pipe, p2, lower=2, upper=6)

# # choose a strategy from: https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-Models
# tuning = RandomSearch(rng=42)

# tuned_iterated_pipe = TunedModel(
#     model=iterated_pipe,
#     range=[r1, r2],
#     tuning=tuning,
#     measures=[brier_loss, auc, accuracy],  #first is used for the optimization but all are stored
#     resampling=StratifiedCV(nfolds=6, rng=42), # this does ~ 85:15 split 6 times
#     acceleration=CPUThreads(),
#     n=40 # define the total number of models to try
# )
# # we are skipping repeats to save time...

# # bind to data and train:
# mach_tuned_iterated_pipe = machine(tuned_iterated_pipe, X, y)
# fit!(mach_tuned_iterated_pipe)

# # summary:
# # - step 1: determine the parameter values thatt optimize the aggregated cross-validation scores
# # - step 2: train optimal model on *all* available data using the learned parameters from above. This is what is used when calling predict.

# rpt2 = report(mach_tuned_iterated_pipe);
# # inspect the best model
# best_booster = rpt2.best_model.model.evo_tree_classifier

# println(
#     "Optimal Hyper-parameters: \n",
#     "\tmax_depth:  ", best_booster.max_depth, "\n",
#     "\tη:          ", best_booster.η, "\n"
# )

# # using our confidence intervals from earlier
# e_best = rpt2.best_history_entry
# confidence_intervals(e_best)

# # we can dig deeper and find the stoping criterion that was used in the optimal model
# rpt2.best_report.controls |> collect

# rpt2.plotting.parameter_names[1] = "η"
# rpt2.plotting.parameter_names[2] = "max depth"

# # visualize the results
# p = plot(mach_tuned_iterated_pipe, size=(600, 450))

# ylabel!(p[1,1], "Brier Loss")
# ylabel!(p[2,2], "Brier Loss")


# # saving our model
# MLJ.save("tuned_iterated_pipe.jls", mach_tuned_iterated_pipe)




# # to gett an even better accurate estimate of performance, we can evaluate our model
# # using stratified cross-validation and all of the data attached to our machine
# e_tuned_iterated_pipe = evaluate(tuned_iterated_pipe, X, y,
#                                  resampling=StratifiedCV(nfolds=6, rng=42),
#                                  measures=[brier_loss, auc, accuracy]
#                                  )

