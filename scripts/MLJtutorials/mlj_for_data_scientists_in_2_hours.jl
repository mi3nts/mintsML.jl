using mintsML
using MLJ
using Plots
using DataFrames

# set the plotting theme
add_mints_theme()
theme(:mints)

# use built-in iris dataset and inspect the features and targets
X_iris, y_iris = @load_iris;

# this tells us the data types and scitypes
schema(X_iris)

y_iris[1:4]  # <-- categorical so we will need some kind of encoding

levels(y_iris)  # <-- get the unique categories


DecisionTree = @load DecisionTreeClassifier pkg=DecisionTree  # <- load model type
model = DecisionTree(min_samples_split=5) # <- instantiate model

# a *model* is just a container for hyper-parameters. It does *not* store the learned parameters.

mach = machine(model, X_iris, y_iris)  # <-- bind model to data

# machine is model + hyper-parameters + learned parameters
# NOTE: observations are rows, not columns
train_rows = vcat(1:60, 91:150);
fit!(mach, rows=train_rows)

fitted_params(mach)

# we can access and mutate the model's parameters. This may cause a retraining
mach.model.min_samples_split = 10
fit!(mach, rows=train_rows)

# make some predictions on another view of the data
predict(mach, rows=71:73)

# or on news data
Xnew = (sepal_length = [5.1, 6.3],
        sepal_width = [3.0, 2.5],
        petal_length = [1.4, 4.9],
        petal_width = [0.3, 1.5]
        )

ŷ = predict(mach, Xnew)


# this prediction is probabilistic. We can get the raw probabilities via:
pdf.(ŷ, "virginica")

# or a single prediction
ŷ[2]




# ----- TELCO dataset -----------------------------------------------------

data = OpenML.load(42178)
df0 = DataFrames.DataFrame(data)

first(df0, 4)

# goal: build + evaluate supervised learning models to predict the `Churn` variable

# Type Coercion
scitype(["cat", "mouse", "dog"])

schema(df0) |> DataFrames.DataFrame # convert schema to df for easier reading

# all `Textual` fields should be either`Multiclass` or `Continuous`


# first let's figure out how to deal with blanks
fix_blanks(v) = map(v) do x
    if x == " "
        return "0.0"
    else
        return x
    end
end

# fix the blanks in this column
df0.TotalCharges = fix_blanks(df0.TotalCharges);

# now we cant to change the scitype to `Continuous`
coerce!(df0, :TotalCharges => Continuous);

# now let's coerce remaining `Textual` into `Multiclass`
coerce!(df0, Textual => Multiclass);

# finally coerce `Churn` into `OrderedFactor` rather than `Multiclass` since to allow metrics like *true positive rate* By convention, first class is the negative one
coerce!(df0, :Churn => OrderedFactor)
levels(df0.Churn)

# re-inspect the scitypes
schema(df0) |> DataFrames.DataFrame


# perform train-test split to generate holdout set for testing
# to speed things up we will toss out 90% of the data and to ad 70/30 split on remainder

df, df_test, df_dumped = partition(df0,
                                   0.07, 0.03,
                                   stratify = df0.Churn, # make sure we maintain target distribution
                                   rng=42  # set the seed for reproducability
                                   )

size(df)
size(df_test)
size(df_dumped)

# now we further split into targets and features
y, X = unpack(df, ==(:Churn), !=(:customerID)); # we skip the customer id category
schema(X)

# do the same for the holdout data
#ytest, Xtest = unpack(df_test, col -> col ∈ [:Churn, :customerID], !=(:customerID))
ytest, Xtest = unpack(df_test, ==(:Churn), !=(:customerID))


# load a model and check its type requirements
Booster = @load EvoTreeClassifier pkg=EvoTrees

# instantiate model with default hyperparameters
booster = Booster()

# check to make sure model will work with our targets
scitype(y) <: target_scitype(booster)

# check to make sure the model will work with our inputs
scitype(X) <: input_scitype(booster)

# this is because the model expects features to be `Continuous`.
# we will add a feature encoding step to ensure that our model will be compatible

# create a linear (non-branching) pipeline
pipe = ContinuousEncoder() |> booster

# pipelines are implementation of more general *model composition* interface
pipe.evo_tree_classifier.max_depth # <-- hyperparameters are nested *and* accessible

# Evaluating the model's performance

# show a complete list of measures
measures()
measures("Brier")

# NOTE: quantileloss is already defined!
QuantileLoss(; τ=0.95)

#  1. evaluate by hand (with a holdout set)
mach_pipe = machine(pipe, X, y)  # attach training data to pipeline in machine
# further split up into train, and validation
train, validation = partition(1:length(y), 0.7)
fit!(mach_pipe, rows=train)

fp = fitted_params(mach_pipe);
keys(fp)

# make sure the encoder did not drop any features
Set(fp.continuous_encoder.features_to_keep) == Set(schema(X).names)

# since this is a tree model, extract feature importances from the report
rpt = report(mach_pipe)
keys(rpt.evo_tree_classifier)
fi = rpt.evo_tree_classifier.feature_importances
feature_importance_table = (feature=Symbol.(first.(fi)), importance=last.(fi)) |> DataFrames.DataFrame


sort!(feature_importance_table, :importance, rev=true)

fi_df = feature_importance_table[1:15, :]

labels = String.(fi_df.feature)
bar(fi_df.importance, yticks=(1:nrow(fi_df), labels), orientation=:horizontal, yflip=true, legend=false,label="")


# for models not supporting feature_importances we should use shapely.jl for model agnostic feature importances. We should do that by default and then if they exist, also do the above ranking.

ŷ = predict(mach_pipe, rows=validation)

print(
    "Measurements:\n",
    "\tbrier loss: ", brier_loss(ŷ, y[validation]) |> mean, "\n",
    "\tauc:        ", auc(ŷ, y[validation]), "\n",
    "\taccuracy:   ", accuracy(mode.(ŷ), y[validation])   # we need mode to generate point predictions
)

# we could have also used `predict_mode` to generate the predictions
# generate confusion matrix and ROC curve

confmat(mode.(ŷ), y[validation])
roc_curve = roc(ŷ, y[validation])

plt = scatter(roc_curve, legend=false)
plot!(plt, xlabel="false positive rate", ylabel="true positive rate")
plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black)




#  2. automated performance evaluation (the *more typical* workflow)

# we will evaluate using a stratified cross validation instead of the holdout validation set
e_pipe = evaluate(
    pipe, X, y,
    resampling = StratifiedCV(nfolds=10, rng=42),
    measures=[brier_loss, auc, accuracy],
    repeats=3,
    acceleration=CPUThreads()
)

# use std on CV folds to estimate confidence interval for performance metrics
# see: https://arxiv.org/abs/2104.00673 for why this may be a bad choice
using Measurements

function confidence_intervals(e)
    factor = 2.0 # to get level of 95%
    measure = e.measure
    nfolds = length(e.per_fold[1])
    measurement = [e.measurement[j] ± factor*std(e.per_fold[j])/sqrt(nfolds - 1)
                   for j in eachindex(measure)]
    table = (measure=measure, measurement=measurement)
    return DataFrames.DataFrame(table)
end

confidence_intervals_basic_model = confidence_intervals(e_pipe)


# ---------------- filtering out unimportant features ------------------------
# let's use our fitted model to update our pipeline
# drop those features with low feature importance to allow us to speed up optimization later
unimportant_features = filter(:importance => <(0.005), feature_importance_table).feature
pipe2 = ContinuousEncoder() |> FeatureSelector(features=unimportant_features, ignore=true) |> booster

# for iterative models (i.e. models that train through a set of steps), it is convenient to set
# a list of controls that define when to stop training. We do this with a wrapper that makes our
# pipe into an 'IteratedModel'

# the list of controls are here: https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided

controls = [
    Step(1),              # to increment iteration parameter (`pipe.nrounds`)
    NumberSinceBest(4),   # main stopping criterion
    TimeLimit(2/3600),    # never train more than 2 sec
    InvalidValue()        # stop if NaN or ±Inf encountered
]

# now wrap our pipeline with these controls into an `IteratedModel`
iterated_pipe = IteratedModel(model=pipe2,
                              controls=controls,
                              measure=brier_loss,
                              resampling=Holdout(fraction_train=0.7)  # i.e. at each training step, evaluate against a holdout fraction of our supplied training data
                              )

mach_iterated_pipe = machine(iterated_pipe, X, y)  # we bind to all data not in test set
fit!(mach_iterated_pipe)

# summary of what's happening:
# - controlled iteration step to tune hyperparameters which uses the holdout set to evaluate performance metric and stops when the control criteria are met
# - a final training step that trains atomic model on *all* of the data using number of iterations determined in step 1.




# Hyper-parameter optimization (model tuning)
# we choose to optimize both `max_depth` and `η`
# just like iteration control, hyperparameter optimization is done with a model wrapper: `TunedModel`
show(iterated_pipe, 2)  # show 2 levels of parameters since it's nested

# define the learnable hyperparameters
p1 = :(model.evo_tree_classifier.η)
p2 = :(model.evo_tree_classifier.max_depth)

# define allowable ranges
r1 = range(iterated_pipe, p1, lower=-2, upper=-0.5, scale=x->10^x)  # make it logarithmic
r2 = range(iterated_pipe, p2, lower=2, upper=6)

# choose a strategy from: https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-Models
tuning = RandomSearch(rng=42)

tuned_iterated_pipe = TunedModel(
    model=iterated_pipe,
    range=[r1, r2],
    tuning=tuning,
    measures=[brier_loss, auc, accuracy],  #first is used for the optimization but all are stored
    resampling=StratifiedCV(nfolds=6, rng=42), # this does ~ 85:15 split 6 times
    acceleration=CPUThreads(),
    n=40 # define the total number of models to try
)
# we are skipping repeats to save time...

# bind to data and train:
mach_tuned_iterated_pipe = machine(tuned_iterated_pipe, X, y)
fit!(mach_tuned_iterated_pipe)

# summary:
# - step 1: determine the parameter values thatt optimize the aggregated cross-validation scores
# - step 2: train optimal model on *all* available data using the learned parameters from above. This is what is used when calling predict.

rpt2 = report(mach_tuned_iterated_pipe);
# inspect the best model
best_booster = rpt2.best_model.model.evo_tree_classifier

println(
    "Optimal Hyper-parameters: \n",
    "\tmax_depth:  ", best_booster.max_depth, "\n",
    "\tη:          ", best_booster.η, "\n"
)

# using our confidence intervals from earlier
e_best = rpt2.best_history_entry
confidence_intervals(e_best)

# we can dig deeper and find the stoping criterion that was used in the optimal model
rpt2.best_report.controls |> collect

rpt2.plotting.parameter_names[1] = "η"
rpt2.plotting.parameter_names[2] = "max depth"

# visualize the results
p = plot(mach_tuned_iterated_pipe, size=(600, 450))

ylabel!(p[1,1], "Brier Loss")
ylabel!(p[2,2], "Brier Loss")


# saving our model
MLJ.save("tuned_iterated_pipe.jls", mach_tuned_iterated_pipe)




# to gett an even better accurate estimate of performance, we can evaluate our model
# using stratified cross-validation and all of the data attached to our machine
e_tuned_iterated_pipe = evaluate(tuned_iterated_pipe, X, y,
                                 resampling=StratifiedCV(nfolds=6, rng=42),
                                 measures=[brier_loss, auc, accuracy]
                                 )

