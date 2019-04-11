# Thinking-Machine

The core idea is to have a complex module that is applied recursively until you obtain a desired confidence. 
After this, you simply apply a final module on the resulted hidden state and return the output.

Basically, this 'complex' module should find the best hidden state for the final module. 
The search, however, is not by applying one big fat network to the input, but by iteratively applying the same 'complex' module over and over again.
