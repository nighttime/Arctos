# Ursa // Arctos
Learning entailment [premise |= hypothesis] by learning to search language model embedding space

### Model Architecture
This model consists of a sentence transformer which encodes both premise and hypothesis, plus an additional MLP which projects the hypothesis encoding somewhere else in embedding space. The model is trained to minimize the distance from this projection to the premise, and maximize the distance to a "null" premise.

### The Name??
Arctos is the "grizzly" species of bear (genus Ursa) 
