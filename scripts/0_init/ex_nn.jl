using Flux

# Create a LeNet model
model = Chain(
    Conv((5,5),1 => 6, relu),
    MaxPool((2,2)),
    Conv((5,5),6 => 16, relu),
    MaxPool((2,2)),
    Flux.flatten,
    Dense(256=>120,relu),
    Dense(120=>84, relu),
    Dense(84=>10, sigmoid),
    softmax
)

# -----------------------------------------------------------------------------


# Function to measure the model accuracy
function accuracy()
    correct = 0
    for index in 1:length(y_test)
        probs = model(Flux.unsqueeze(x_test[:,:,:,index],dims=4))
        predicted_digit = argmax(probs)[1]-1
        if predicted_digit == y_test[index]
            correct +=1
        end
    end
    return correct/length(y_test)
end

# Reshape the data
x_train = reshape(x_train, 28, 28, 1, :)
x_test = reshape(x_test, 28, 28, 1, :)

# Assemble the training data
train_data = Flux.DataLoader((x_train,y_train), shuffle=true)

# Initialize the ADAM optimizer with default settings
optimizer = Flux.setup(Adam(), model)

# Define the loss function that uses the cross-entropy to
# measure the error by comparing model predictions of
# data row "x" with true data from label "y"
function loss(model, x, y)
	return Flux.crossentropy(model(x),Flux.onehotbatch(y,0:9))
end

# Train model 10 times in a loop
for epoch in 1:10
    Flux.train!(loss, model, train_data, optimizer)
    println(accuracy())
end
