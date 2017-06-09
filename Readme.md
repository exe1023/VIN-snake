# Use Value Iteration Network to play snake

## Train:
`make train ATTENTION=$attention`

## Play:
###Command line
`make play MODEL=$modelpath ATTENTION=$attention`

###GUI
`make play-gui MODEL=$modelpath ATTENTION=$attention`

##Attention module:

### Attention -1:
The baseline DQN model.
### Attention 0:
Hard attention which takes the position of the snake head.
### Attention 1:
Conv layer on the input image.
### Attention 2:
Fully connected layer on the input image.
### Attention 3:
Conv layer on the stacked input and value images.

## Code Ref:
<a href="https://github.com/YuriyGuts/snake-ai-reinforcement">Snake Environment and DQN</a>

<a href="https://github.com/neka-nat/vin-keras">Keras Implementation of Value Iteration Network</a>
