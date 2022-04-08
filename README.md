# Artificial-Neural-Network-MNIST
# Artificial-Neural-Network-MNIST-Java
This is an artificial neural network that is coded by me using pure Java (that can be trained to recognise handwritten digits). The code is somewhat messy and poorly documented as I created this project
in order to learn. It features a neural network library I coded which is adapted to funcion specifically for the MNIST dataset. With minor alterations,
this could be used for any task in which an artificial neural network is used. The iteration featured here makes use of the ADAM optimisation technique and uses the reLU
activation function for all layers bar the final layer (where a softmax activation is used). All weigths and biases are initialised using the method proposed by He et Al. in 2015 (commonly refered to as He initilaisation). My actual code for the neural network is functional and works well, however there is no doubt that it is messy. I am
releasing this project to allow people to come to grips with how the tuning of hyperparameters can impact network performance. 
In order to tume the hyperparameters, one should select HandWrittenDigitMNSIT class. The size of the network is set in the constrcutor NeuralNetwork ne = new NeuralNetwork
(new int[] {784,128,10} ); Each element in the array represents the number of nodes in a given layer. The number of elements represents the number of layers. To see
how networks of different sizes perform in terms of accuracy and speed, alter this array. However, the array's first element (i.e. element 0) must always be 784 and the last element must always
be 10. Altering the learning rate will impact the size of the step taken by the network once the gradient has been calculated. Altering this should help provide an intuitive feel
for how higher learning rates may lead to faster but ultimately less accurate convergence. This is set to 0.001 by default. The variable noOfEpochs represents the number of complete
passes the network makes on the training data. E.g. noOfEpochs = 2 would mean that the network sees each training record twice across the training phase. I have set this value to 5 as it leads to a fast yet accurate convergence. The variable storing the size of the training data should not be changed as it indicates the size of the training data,
 which in this case is 60,000 (with the test data having 10,000 handwritten images). The variable batchSize stores how many items handwritten digits should be analysed
  per updating of parameters. E.g. if this value is set to one, one image will be used to calculate the total loss and then this loss will be used to update the parameters.
  If the batch size is greater, each step will be more accurate although it will also be more computationally expensive. If the batch size is set to the size of the data set, it will
   be slow but also lead to a higher chance of the network getting stuck at a local minimum. I have set the batch size to 128 by defualt. The variable 
   printAfterNoIterations determines how often the accuracy of the system should be tested and displayed. If a network has a batchSize of one, the system will
   perform 60,000 iterations per epoch (i.e. the size of the training data). If the batch size is 2, the system will perform 30,000 iterations per epoch. The lower the print value
   is, the more often the system is tested. However, if this value is too low it may cause the system to become too slow as testing takes a few seconds (as it tests the 
   network's performance on the 10,000 images in the training set). This value may need the most tuning and depends on the sample size. The
   variable loadDataInMemory determines whether the test data is stored in the system's memory. It is highly recommeneded that this variable is set to true as the
   performance degrades significantly when the data is not stored in memory. This option is here as my family's desktop computer does not have enough memory to execute the network
   without this variable being false (however my laptop, which I coded the network on, does).
   Once the network finishes training, it opens up a visualisation tool which displays the accuracy of the network and shows how it performs for every image on the training
   set.
   There is also a JAR file that has 98.99% percent acccury, pretrained system.
   This project was made using NetBeans IDE 8.2.
