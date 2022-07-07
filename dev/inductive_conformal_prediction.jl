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

z = x.^5 .+ y.^4 .- x.^4 .- y.^3 ;


Xtrain = X[:, 1:1500];
Ytrain = z[1:1500];

Xcal = X[:, 1501:1750];
Ycal = z[1501:1750];

Xtest = X[:, 1751:end];
Ytest = z[1751:end];

println(size(Xtrain), " ", size(Xcal), " ", size(Xtest))



p = plot(Xtrain[1,:], Xtrain[2,:], Ytrain, seriestype = :scatter, color=:blue, ms = 2,label="train", camera=(60, 25), xlabel="x", ylabel="y", zlabel="f(x,y)")
plot!(p, Xcal[1,:], Xcal[2,:], Ycal, seriestype = :scatter, color=:red, ms=2, label="calibration")
plot!(p, Xtest[1,:], Xtest[2,:], Ytest, seriestype = :scatter, color=:gray, ms=2, label="test")

display(p)


# create the NN model

model = Chain(
    Dense(2 => 3, σ),
    Dense(3 => 3)  # 0.05, 0.5, 0.95 quartiles
)

# test it out
model(Xtest[:,1:10])



# define the loss function
function ℓ_ϵ(y, ŷ, ϵ)
    return maximum([ϵ*(y-ŷ), (1-ϵ)*(ŷ-y)])
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


# NOTE: we need to add summation to these loss functions. 


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


# set up optimizer
optimizer = ADAM()




training_loss = Float64[]
epochs = Int64[]


for epoch in 1:10
	  Flux.train!(Loss, params(model), zip(Xtrain, Ytrain), optimizer)

	  if epoch % 10 == 0
	      # we record our training loss
		    push!(epochs, epoch)
		    push!(training_loss, Loss(Xtrain, Ytrain))
	  end
end


