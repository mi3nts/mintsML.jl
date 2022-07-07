using Plots, LaTeXStrings
using Statistics
using Distributions
using Flux

# create Dataset


# see this paper: https://arxiv.org/pdf/1909.12122.pdf

n_train= 2000;

X = 2 .* (rand(2,n_train) .- 0.5);

x = @view X[1,:]
y = @view X[2,:]

target_func(x,y) = x^5 + y^4 -x^4 - y^3
z = target_func.(x,y)


Xtrain = X[:, 1:1500];
Ytrain = z[1:1500]';

Xcal = X[:, 1501:1750];
Ycal = z[1501:1750]';

Xtest = X[:, 1751:end];
Ytest = z[1751:end]';



p = plot(Xtrain[1,:], Xtrain[2,:], Ytrain', seriestype = :scatter, color=:blue, ms = 2,label="train", camera=(60, 25), xlabel="x", ylabel="y", zlabel="f(x,y)")
plot!(p, Xcal[1,:], Xcal[2,:], Ycal', seriestype = :scatter, color=:lightgreen, ms=2, label="calibration")
plot!(p, Xtest[1,:], Xtest[2,:], Ytest', seriestype = :scatter, color=:gray, ms=2, label="test")

display(p)

savefig("figures/dataset.svg")

# create the NN model

model = Chain(
    Dense(2 => 10, relu),
    Dense(10 => 10, relu),
    Dense(10 => 3)  # 0.05, 0.5, 0.95 quartiles
)

# test it out
model(Xtest[:,1:10])

y = rand(1, 10)
ŷ = rand(1, 10)
ϵ = 0.05

# define the loss function
function ℓ_ϵ(y, ŷ, ϵ; agg=mean)
    u = y - ŷ
    agg([maximum([ϵ*uᵢ, (ϵ-1)*uᵢ]) for uᵢ ∈ u])
end


# visualize the loss function for a set of ϵ
ys = -1:0.1:1
L_05 = ℓ_ϵ.(0, ys, 0.05)
L_5 = ℓ_ϵ.(0, ys, 0.5)
L_95 = ℓ_ϵ.(0, ys, 0.95)

plot(ys, L_05, linestyle=:dash, color=:lightblue, label=L"\epsilon=0.05", guidefontsize=15)
plot!(ys, L_5, color=:black, label=L"\epsilon=0.5")
plot!(ys, L_95, linestyle=:dash, color=:darkblue, label=L"\epsilon=0.95")
xlabel!(L"y")
ylabel!(L"\ell_\epsilon")
title!(L"Pinball Loss with $\hat y = 0$")


savefig("figures/pinball.svg")

α = 0.10
# now we need to code this up
function ℓ_1(X,y)
    ŷ = model(X)[1,:]
    return sum(ℓ_ϵ(y, ŷ, α/2))
end

function ℓ_2(X,y)
    ŷ = model(X)[2,:]
    return sum(ℓ_ϵ(y, ŷ, 0.5))
end

function ℓ_3(X,y)
    ŷ = model(X)[3,:]
    return sum(ℓ_ϵ(y, ŷ, 1.0 - α/2))
end

ℓ_1(Xtrain[:, 1:10], Ytrain[1:10])
ℓ_2(Xtrain[:, 1:10], Ytrain[1:10])
ℓ_3(Xtrain[:, 1:10], Ytrain[1:10])

function Loss(X,y)
    return ℓ_1(X,y) .+ ℓ_2(X,y) .+ ℓ_3(X,y)
end


Loss(Xtrain[:, 1:10], Ytrain[1:10])


# add non-crossover penalty
function Penalty(X)
    Ypred = model(X)
    y_05 = @view Ypred[1,:]
    y_5 = @view Ypred[2,:]
    y_95 = @view Ypred[3,:]

    # so we should have y_05  <= y_5 <- y_95
    Δ1 = y_05 .- y_5
    Δ2 = y_5 .- y_95

    penalty1 = sum([maximum([0, Δ]) for Δ ∈ Δ1].^2)
    penalty2 = sum([maximum([0, Δ]) for Δ ∈ Δ2].^2)

    return penalty1 + penalty2
end
# check that it works
Penalty(Xtrain)

# define total loss
ΣLoss(X,y) = Loss(X,y) + 0.5*Penalty(X)
# check that it works
ΣLoss(Xtrain, Ytrain')


# set up optimizer
optimizer = ADAM()


training_loss = Float64[]

function cb()
		push!(training_loss, sum(Loss(Xtest, Ytest')))
end



# construct batches if desired
data = Flux.DataLoader((Xtrain, Ytrain'), shuffle=true, batchsize=32)

n_epochs = 50

# Flux.train!(Loss, Flux.params(model), [(Xtrain, Ytrain')], optimizer)
Flux.@epochs n_epochs Flux.train!(ΣLoss, Flux.params(model), data, optimizer, cb=cb)




p1 = plot(training_loss, xlabel="batch #", ylabel="Loss", title="Training", lw=3, label="")


preds_train = model(Xtrain)[2,:]
preds_test = model(Xtest)[2,:]

p2 = scatter(Ytrain', preds_train, color=:lightblue, ms=2, msw=0, alpha=0.7, label="training", legend=:topleft)
xlabel!("truth")
ylabel!("model")
plot!(Ytest', preds_test, seriestype=:scatter, ms=2, msw=0, alpha=0.7, color=:lightgreen, label="test")


P1 = plot(p1, p2, layout=(1,2))


# get the quartile correction
function q̂(X,y, α)
    # 1. compute non-conformity scores S(x,y)
    Ypred = model(X)
    q_low = @view Ypred[1,:]
    q_high = @view Ypred[3,:]

    Δlow = q_low .- y'
    Δhigh = y' .- q_high

    s = [maximum([Δlow[i], Δhigh[i]]) for i ∈ 1:size(y,2)]

    n = size(y, 2)
    return quantile(s, ceil((n+1)*(1-α))/n )
end


q = q̂(Xcal, Ycal, α)


function ICP(X,q̂)
    Ypred = model(X)
    ŷ_low = Ypred[1,:]'
    ŷ_high = Ypred[3,:]'

    ci_low = ŷ_low .- q̂
    ci_high = ŷ_high .- q̂

    return vcat(ci_low, ci_high)
end


# test it out!
ICP(Xtest, q)



# let's test it out on one slice
x = -1:0.1:1
y = zeros(size(x))

Xtry = vcat(x',y')

z_true = target_func.(x,y)

Z_model = model(Xtry)
ICP_model = ICP(Xtry, q)


p3 = plot(x, ICP_model[1,:], fillrange= ICP_model[2,:], fillalpha=0.75, color=nothing, fillcolor=:blue, label = "90% ICP")
plot!(x, Z_model[1,:], fillrange = Z_model[3,:] , fillalpha = 0.75, color=nothing, fillcolor=:gray, label = "90% Confidence band")
xlabel!("x")
ylabel!("z")
title!("Example Prediction")
plot!(x, Z_model[2,:], color=:blue, label="prediction")
scatter!(x, z_true, color=:red, ms=3, label="True Value", legend=:bottomright)

p4 = plot(P1, p3, layout=(2,1))


savefig("figures/ICP_res.svg")

display(p4)
