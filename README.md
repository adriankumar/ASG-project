# ASG-project
Attempting augmentative speech generation using BCI 

currently working on a demo/proof of concept before moving on to real bci usage
demo is using 3 samples from Chisco dataset of imagined speech

to do (soon):
- Start writing some documentation
- Fix substring reward hacking
- fix decoding to be purely sampled instead of argmax AND sampling to prevent different reward and penalty signals and align infernence capabilities
- Change the training batch of eeg data to be semantically related so that gradients can properly converge to semantic patterns; currently each batch is randomised imagined speech samples, making the learning extremely difficult, especially with gpt2's large embedding space that it needs to learn
- Fix/debug reward and penalty schemes and figure out better way to handle weightings
