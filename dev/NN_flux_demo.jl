using Plots, LaTeXStrings
using Flux

# create Dataset


n_train= 2000;

X = 2 .* (rand(2,n_train) .- 0.5);

x = @view X[1,:]
y = @view X[2,:]

z = x.^5 .+ y.^4 .- x.^4 .- y.^3 ;


Xtrain = X[:, 1:1500];
Ytrain = z[1:1500];

Xtest = X[:, 1501:end];
Ytest = z[1501:end];




p = plot(Xtrain[1,:], Xtrain[2,:], Ytrain, seriestype = :scatter, color=:blue, ms = 2,label="train", camera=(60, 25), xlabel="x", ylabel="y", zlabel="f(x,y)")
plot!(p, Xtest[1,:], Xtest[2,:], Ytest, seriestype = :scatter, color=:gray, ms=2, label="test")

display(p)


# create the NN model

model = Chain(
    Dense(2 => 10, relu),
    Dense(10 => 10, relu),
    Dense(10 => 1)  # 0.05, 0.5, 0.95 quartiles
)


Flux.params(model)

# test it out
model(Xtest[:,1:10])


# define the loss function

Loss(X, y) =  Flux.mse(model(X), y)


# try it out
size(Xtrain)
size(Ytrain)
size(model(Xtrain))

Loss(Xtrain[:,1], Ytrain[1])


# set up optimizer
optimizer = ADAM()


training_loss = Float64[]

function cb()
		push!(training_loss, sum(Loss(Xtest, Ytest')))
end



# construct batches if desired
data = Flux.DataLoader((Xtrain, Ytrain'), shuffle=true, batchsize=32)

n_epochs = 30

# Flux.train!(Loss, Flux.params(model), [(Xtrain, Ytrain')], optimizer)
Flux.@epochs n_epochs Flux.train!(Loss, Flux.params(model), data, optimizer, cb=cb)



p1 = plot(training_loss, xlabel="batch #", ylabel="MSE loss")


preds_train = model(Xtrain)
preds_test = model(Xtest)

p2 = plot(Ytrain, preds_train', seriestype=:scatter, color=:lightblue, alpha=0.7, label="training")
xlabel!("truth")
ylabel!("model")
plot!(Ytest, preds_test', seriestype=:scatter, color=:lightgreen, label="test")


plot(p1, p2, layout=(1,2))



