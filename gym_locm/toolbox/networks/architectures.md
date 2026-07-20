# Gym-LOCM Network Architectures

This document provides a high-level overview of the neural network architectures implemented in `gym_locm/toolbox/networks/` for learning the Legends of Code and Magic (LOCM) environment. 

The architectures represent a steady evolution from rigid flat vectors to highly relational, joint-attention set transformers.

---

## 1. Simple Network (`simple.py`)
The baseline approach. It takes hand-crafted, flattened features representing the game state (e.g., player stats, concatenated card stats) and passes them through standard Multi-Layer Perceptrons (MLPs).
- **Pros**: Computationally cheap and easy to train.
- **Cons**: Extremely rigid. It does not exploit the permutation invariance of cards in a hand or creatures on a board. If you shuffle the cards in your hand, the network sees a completely different input vector and has to learn from scratch that the state is effectively the same.

## 2. Deep Sets (`deep_sets.py`)
Based on the Deep Sets architecture (Zaheer et al.). It introduces permutation invariance by treating collections of items (like a hand of cards) as unordered sets.
- **Mechanism**: Each entity (card or creature) is processed independently by a shared MLP ($\phi$). The outputs are then pooled together using an order-invariant operation (like sum, mean, or max). The pooled result is passed through another MLP ($\rho$).
- **Pros**: Perfectly permutation invariant. Scales easily to arbitrary set sizes.
- **Cons**: Lacks relational capacity. Entities cannot interact with each other *before* pooling. A card in your hand cannot contextualize its value based on another card in your hand until after they are both squashed into a single global pool.

---

## The Set Transformer Evolution
To solve the relational bottleneck of Deep Sets, the Set Transformer was introduced, leveraging Self-Attention to allow entities to communicate. All versions have been upgraded to use highly optimized PyTorch native `nn.TransformerEncoderLayer` blocks (with FlashAttention and LayerNorm built-in).

### V1: The "Hierarchical Pooled" Set Transformer (`set_transformer.py`)
- **Mechanism**: Processes each game zone (deck, hand, player lanes, opponent lanes) independently through its own Transformer blocks. It then immediately pools these zones down into single vectors using Pooling by Multihead Attention (PMA). The pooled vectors are concatenated to form the global state.
- **Pros**: Elements within the *same* zone (e.g., cards in hand) can now attend to each other.
- **Cons**: Cross-zone interaction is limited. A card in your hand cannot directly attend to a creature on the board because they are pooled independently before being combined.

### V2: "Joint Global Attention" (`set_transformer_2.py`)
- **Mechanism**: Removes early pooling. It concatenates almost all individual entities (unpooled hand cards, unpooled lane creatures, players, pooled deck) into one massive, heterogeneous "Set" (e.g., 23 elements). To help the network distinguish between them, a hard-coded 8-dimensional one-hot vector (e.g., "I am an enemy creature") is concatenated to each entity. This massive set is run through deep global Transformer blocks.
- **Pros**: Massive leap in representational power. Entities can now cross-attend directly (a spell in hand can calculate its synergy with a specific creature on the board).
- **Cons**: The concatenated one-hot vectors increase the dimensionality and are somewhat rigid.

### V2.1: "Refined Joint Attention" (`set_transformer_2_1.py`)
- **Mechanism**: Takes V2 and applies modern Transformer best practices.
  - **Learnable Positional/Type Encodings**: Replaces the concatenated one-hot vectors with learnable additive embeddings (similar to positional encodings in NLP). These are added directly to the features, keeping dimensionality lean.
  - **Richer Deck Representation**: The deck is pooled into 4 vectors (instead of 1) to extract richer summary statistics.
  - **Capacity**: Scales hidden dimension from 64 to 128.
- **Pros**: More flexible, stable, and expressive than V2.

### V2.2: "Action-Specific Query Vectors" (`set_transformer_2_2.py`)
- **Mechanism**: Takes V2.1 and revolutionizes how the action heads operate. In earlier versions, dense layers statically pulled information out of the entities to predict actions. V2.2 introduces Action-Specific Query Vectors using `PMA` for the `summon`, `use`, and `attack` actions. 
- **How it works**: A learnable query embedding tailored for a specific action dynamically cross-attends to the entire `all_entities` set. It extracts exactly the contextual information needed for that specific action, producing a dynamic context vector that is added to the logit calculations.
- **Pros**: Action heads are no longer blind dense layers; they actively search the game state for the information they need to execute an action.
