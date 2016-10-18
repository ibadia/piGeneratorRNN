# piGeneratorRNN

This is a small experiment in which i tried to generate the suceeding value of pi using the preeceding values.
E.g: pi=3.14159265359........

We can easily generate the dataset from this using this simple logic
Query   Result
141     5
415     9
159     2
592     6
926     5

We can easily generate a dataset from this.
This is done using the 1 billion pi values at https://stuff.mit.edu/afs/sipb/contrib/pi/

We can make the query bigger in a wish to obtain the logic.


The file pi_addition_rnn.py is just the modification of https://github.com/fchollet/keras/blob/master/examples/addition_rnn.py
I modified this python code to make the same RNN in hope find the pattern in the value of pi but the model on this particular set seems to overfit the data (gives 100% accuracy on training, very less accuracy on testing data).
