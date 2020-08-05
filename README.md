# Thinking-Machine

The core idea is to have a complex module that is applied recursively until you obtain a desired confidence. 
After this, you simply apply a final module on the resulted hidden state and return the output.

Basically, this 'complex' module should find the best hidden state for the final module. 
The search, however, is not by applying one big fat network to the input, but by iteratively applying the same 'complex' module over and over again.


# Thinking-Machine abstract

The core idea is to have a complex module that is applied recursively until you obtain a desired confidence.
After this, you simply apply a final module on the resulted hidden state and return the output.

Basically, this 'complex' module should find the best hidden state for the final module.
The search, however, is not by applying one big fat network to the input,
but by iteratively applying the same 'complex' module over and over again.

# Main components of this 'complex' module

## Final Classifier:  
this will take the last hidden state and return the final output/label.

## Confidence Evaluator:  
this will output the confidence with which the Final Classifier can be applied at the current hidden state

## Query Machine:  
this will look at the current state (and maybe also a hidden state of the Confidence Evaluator) and output  
a query

## Answer Machine:
this will take the current state and the query  and output some values which will be summed to the current state, in   
this way obtaining the next state


