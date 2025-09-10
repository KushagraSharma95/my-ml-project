# NeuroAI at a Crossroads: A Critical Appraisal of Neuroscience‑Inspired AI and AI‑Inspired Neuroscience

## Abstract
NeuroAI—the bidirectional exchange between neuroscience and artificial intelligence—promises to make learning machines more data‑efficient, robust, and interpretable while offering the brain sciences powerful computational theories. Yet the field’s rhetoric often outruns its evidence. This critique examines NeuroAI’s central claims through the lenses of technical feasibility, modeling assumptions, and scientific utility. We argue that (i) current “brain‑like” systems capture narrow slices of cortical function but remain brittle outside their training regimes; (ii) many biologically motivated learning rules trade tractability for plausibility without yet delivering clear performance or explanatory wins; (iii) architectural inspiration from the brain (recurrence, modulatory control, dendritic computation, spikes) is most valuable when formalized as constraints in optimization, not as wholesale imitation; and (iv) evaluation should prioritize jointly satisfying behavioral performance, neural/behavioral alignment, **and** engineering metrics such as energy, latency, and adaptability. We propose concrete, testable benchmarks and a staged research program that integrates mechanistic neuroscience with machine learning practice. The upshot is neither “the brain is a blueprint” nor “ignore biology,” but a pragmatic middle road: use neuroscience to **shape the hypothesis space** (objectives, inductive biases, credit‑assignment regimes, hardware constraints) while letting optimization and empirical testing decide what survives.

---

## 1. Introduction
Neuroscience and artificial intelligence have long been intellectual siblings. Classic connectionism treated neural networks as simplified models of cortical circuits; deep learning then matured by largely decoupling from biology, optimizing end‑to‑end objectives at scale. Today, **NeuroAI** seeks a reunification on two fronts: *neuroscience‑for‑AI* (leveraging biological principles to improve learning systems) and *AI‑for‑neuroscience* (using modern models and tools to explain brains). The motivation is compelling. Brains learn from sparse feedback, operate with remarkable energy efficiency, generalize from few examples, and remain robust in open‑ended environments. If machines could absorb even a fraction of these capabilities, the impact on science and society would be profound.

And yet, claims about “brain‑like” models can be slippery. A convolutional net that matches ventral‑stream responses on one dataset may fall apart under distribution shift. A spiking neural network may be energy‑frugal in principle but hard to train in practice. Conversely, the brain’s own mechanisms are heterogenous, multi‑scale, and task‑dependent; any one‑to‑one mapping to machine modules risks oversimplification. The goal of this paper is to **separate aspiration from evidence** and chart a pragmatic path for NeuroAI as an engineering science.

We proceed in six parts. Section 2 clarifies terminology and scope. Section 3 evaluates representational and behavioral alignment between modern AI systems and brains. Section 4 critiques biologically motivated learning rules and credit‑assignment strategies. Section 5 examines architectural and dynamical inspirations—recurrence, attention, modulatory control, dendrites, and spikes. Section 6 discusses objectives and self‑supervision. Section 7 proposes evaluation principles and benchmarks that integrate scientific and engineering desiderata. Section 8 addresses neuromorphic hardware. Section 9 identifies methodological pitfalls and ethical questions. Section 10 outlines a staged research agenda, followed by a brief conclusion.

---

## 2. Scope, Terms, and a Working Definition
**NeuroAI** is often used to describe anything at the intersection of brains and machines, but ambiguity hinders progress. We adopt the following working definition:

- **Neuroscience‑for‑AI (N4AI):** Extract constraints and inductive biases from biology—objective functions, learning rules, architectures, and resource constraints—to improve machine performance or efficiency.
- **AI‑for‑neuroscience (AI4N):** Use machine learning models as *explanatory tools* for neural data, behavior, and cognition, including mechanistic models that make falsifiable predictions.

These halves should be symbiotic. A model that performs well but fails to capture neural/behavioral phenomena offers limited scientific value; a model that fits neural data but fails on tasks provides limited engineering value. NeuroAI succeeds when **the same mechanistic ingredients** help on both fronts.

We also distinguish three levels of analysis that must be explicitly linked in any NeuroAI claim:
1. **Computational level (objectives):** What is the organism/system optimizing (predictive coding, empowerment, reward, homeostasis, compression)?
2. **Algorithmic level (representations/learning):** What algorithms, signals, and credit‑assignment strategies could achieve it (e.g., error feedback via modulatory pathways, local learning, predictive targets)?
3. **Implementation level (hardware/biophysics):** How are these instantiated (dendrites, spikes, synaptic plasticity, neuromodulators, glia, energy budgets)?

Vague appeals to “biologically plausible” fall short unless tied to one or more of these levels with precise, testable statements.

---

## 3. Representational and Behavioral Alignment: What Has Been Achieved?
### 3.1 Alignment successes
Deep convolutional networks trained on object recognition tasks can predict variance in primate ventral‑stream responses and human behavior in rapid categorization. Transformers trained on language model next‑token prediction develop internal representations predictive of human lexical and neural signals. Such **task‑driven alignment** suggests that optimizing ecologically relevant objectives can yield features similar to those measured in brains—even without explicit biological constraints.

### 3.2 Where alignment breaks
Despite these successes, several gaps remain:

- **Out‑of‑distribution fragility:** Models aligned on in‑distribution stimuli often fail on atypical poses, corruptions, or novel contexts that humans handle gracefully. This indicates that alignment may be *curve‑fitting* to the dataset rather than capturing generative structure of the environment.
- **Compositional generalization:** Brains flexibly recombine known parts into novel wholes (systematicity). Many neural nets struggle with such generalization without extensive curriculum or architectural bias.
- **Temporal credit and planning:** Many benchmarks assess static recognition, whereas animals excel at credit assignment across long horizons, model‑based planning, and causal reasoning under uncertainty.
- **Multi‑task and continual learning:** Biological agents learn myriad tasks lifelong without catastrophic forgetting, while many machine models require replay, special regularizers, or task‑segmented training.
- **Energy and latency:** Functional alignment that ignores energy cost, latency, and robustness is incomplete; brains meet strict metabolic constraints that current models routinely exceed by orders of magnitude.

### 3.3 What counts as evidence?
Correlating layer activations with neural recordings is a starting point, not an end. Robust evidence requires *convergent validity* across axes:
- **Predictive:** Does the model forecast neural/behavioral responses to **novel** stimuli and tasks?
- **Causal:** Do targeted interventions (lesions, neuromodulation, attention shifts) produce matched changes in the model?
- **Mechanistic:** Are the computations localized and composable in ways that map onto identified circuits?
- **Resource‑aware:** Does alignment persist when energy, noise, and hardware constraints are respected?

Without these, representational alignment risks Goodhart’s law: once a measure becomes a target, it ceases to be a good measure.

---

## 4. Learning Rules and Credit Assignment: Plausible ≠ Practical
### 4.1 The backpropagation debate
Backpropagation through time (BPTT) underlies modern deep learning but is hard to reconcile with biology’s locality and timing. Alternatives—feedback alignment, predictive coding, equilibrium propagation, three‑factor rules—aim to deliver **local credit signals**. These lines of work are scientifically fertile, yet face practical hurdles:
- **Scaling:** Many biologically motivated rules degrade on large, deep, or highly non‑linear tasks without additional scaffolding.
- **Stability:** Recurrent and continuous‑time dynamics can be fragile, demanding careful initialization and tuning.
- **Information routing:** Biologically local updates still require *useful* error information. How is rich credit routed via neuromodulators, dendrites, or oscillations without collapsing to backprop in disguise?

### 4.2 What biology actually provides
Biology offers at least four actionable constraints:
1. **Heterogeneous learning signals:** Dopamine predicts reward prediction error in basal ganglia; acetylcholine and noradrenaline gate uncertainty and exploration; cortical circuits likely combine multiple objectives (predictive, reconstructive, reward‑driven).
2. **Multi‑timescale plasticity:** Synapses adjust on milliseconds to days; structural plasticity and neuromodulation add slower channels. This motivates *hierarchical learners* where fast adapters sit atop slowly adapting priors.
3. **Local recurrence and dendrites:** Dendritic compartments compute non‑linear subunits; recurrence supports attractors and working memory. These suggest *architectural priors* for context and stability rather than direct substitutes for backprop.
4. **Energy and precision limits:** Noise, latency, and metabolic budgets bound what credit‑assignment schemes are plausible; methods that thrive under low‑precision, event‑driven regimes may be favored.

### 4.3 A pragmatic stance
Rather than replacing backprop wholesale, treat neuroscience as **regularization**: it narrows the search over models and training regimes. The live question is not “Is backprop biologically plausible?” but **“Which constraints borrowed from biology improve generalization, efficiency, and robustness at scale?”** Evidence so far points to the utility of self‑supervised objectives, recurrence for context, structured sparsity, modulatory gating, and multi‑timescale adaptation—each of which has biological echoes.

---

## 5. Architectures and Dynamics: Inspiration with Discipline
### 5.1 Recurrence, attention, and gating
Brains are deeply recurrent; sensory cortex integrates over time, and attention is realized through competitive and modulatory circuits. Transformers simulate some of this via attention and gating. The critique is not that transformers are “un‑brain‑like,” but that **explicit temporal credit and control** remain underdeveloped. Incorporating lightweight recurrence, adaptive computation time, and learned controllers can bridge the gap between static feedforward computation and dynamic, resource‑aware processing.

### 5.2 Dendrites and compartmentalization
Dendrites perform non‑linear integration, effectively endowing a neuron with a small network inside. In AI terms, this argues for **intra‑unit structure**: mixture‑of‑experts, gated sub‑modules, and conditional computation that increase capacity without proportional runtime cost. The danger is overfitting the metaphor; dendritic analogs should be adopted when they **win on ablations** (accuracy, efficiency, interpretability), not simply for anatomical rhyme.

### 5.3 Spikes and event‑driven computation
Spiking neurons offer sparse, asynchronous communication aligned with neuromorphic hardware. But three problems persist: (i) training spiking networks remains cumbersome (surrogate gradients, ANN‑to‑SNN conversion), (ii) performance on large‑scale tasks still trails dense ANNs unless carefully engineered, and (iii) software/hardware tooling is fragmented. A measured path is to use spikes where they provide **strict advantages**—low‑power sensing, low‑latency reflexes, or edge deployment—while keeping dense networks where training ecosystems are mature.

### 5.4 Continual and modular computation
Brains are modular yet interactive. Architectural motifs that support **task modularity with shared primitives**—adapters, modular routing, neural module networks—mirror cortical specialization and may reduce interference in continual learning. The core hypothesis is that **right‑sized modularity** improves sample efficiency and transfer, particularly under constrained compute and non‑stationary data.

---

## 6. Objectives and Self‑Supervision: What Is Being Optimized?
Brains likely juggle multiple partially aligned objectives: prediction, compression, control, homeostasis, social reward. Modern AI’s move to **self‑supervised learning (SSL)**—contrastive, masked modeling, generative pretraining—implicitly adopts prediction/compression as central objectives, with reinforcement learning (RL) handling control. This echoes sensorimotor development: pretrain on the world, then fine‑tune by interacting with it.

Two critiques apply. First, current SSL is **disembodied**: models learn from static internet corpora divorced from action and consequence. Second, objectives are often **proxy‑driven** (next token, next frame) without explicit pressure for causal abstraction. NeuroAI can contribute by **closing the loop**—learning objectives that couple perception and action (active inference, empowerment, curiosity), and by aligning SSL targets with **predictive variables brains care about** (object identity, affordances, social signals). Progress here will be measured not just by benchmark scores, but by faster adaptation, safer exploration, and more causal generalization in embodied settings.

---

## 7. Evaluation: From Siloed Metrics to Joint Criteria
NeuroAI requires **joint evaluation** along four axes:
1. **Task performance:** Accuracy, reward, calibration, and robustness on ecologically valid tasks.
2. **Neural/behavioral alignment:** Predictivity of neural responses and human/animal behavior under new stimuli or perturbations.
3. **Mechanistic clarity:** Causal interpretability—can we identify circuits, variables, and interventions that explain behavior?
4. **Engineering metrics:** Energy per inference, latency, memory footprint, precision tolerance, and adaptability.

We propose the **ELSA score**—*E*nergy, *L*atency, *S*ample efficiency, *A*lignment—as a minimal composite metric. A model that wins on alignment but fails energy/latency may be a good scientific model but a poor engineering solution; the inverse also holds. ELSA encourages *pareto‑optimal* progress rather than single‑number heroics.

### 7.1 Benchmarks to make failure visible
- **Embodied OOD:** Train agents in simulated households; test on rearranged objects, novel lighting, and shifted dynamics. Measure success, adaptation speed, and energy use.
- **Cross‑species generalization:** Fit a shared model to human and non‑human primate visual data; test whether the same representation supports both species on new stimuli.
- **Causal perturbation tests:** Predict the effect of targeted lesions/optogenetic‑like parameter ablations on behavior; validate against neural perturbation data where available.
- **Rapid few‑shot learning:** After pretraining, measure under strict data/time budgets whether models can learn new categories or tasks like animals do (dozens, not millions, of labeled samples).
- **Noisy, low‑precision operation:** Evaluate under quantization, event‑driven input, and injected noise to approximate biological constraints.

### 7.2 Reporting standards
Papers should report energy/use metrics (joules per inference, training FLOPs), ablation studies for each biological constraint used, and **negative results** where plausible rules failed. NeuroAI will mature faster when *what does not work* is documented with the same care as successes.

---

## 8. Neuromorphic Hardware: Promise and Caveats
Neuromorphic platforms—digital (e.g., many‑core asynchronous), mixed‑signal, and analog—leverage sparse, event‑driven computation to reduce energy and latency. Spiking hardware promises μJ‑to‑nJ inference at edge power budgets; analog crossbar arrays offer in‑memory multiply–accumulate with orders‑of‑magnitude lower energy than von Neumann architectures. These trends align with biological constraints and make NeuroAI an engineering discipline, not just metaphor.

Caveats are equally real. Analog variability, device drift, limited precision, and programming ecosystems complicate deployment. ANN‑to‑SNN conversion narrows model classes; surrogate‑gradient training on hardware is still nascent; non‑idealities (IR‑drop, device mismatch) must be absorbed at the algorithmic level. A **codesign** mindset—objectives, models, and hardware evolving together—is essential. The right question is not “Can we exactly mimic cortex?” but “Can we design **resource‑aware learners** that exploit sparsity, locality, and event‑driven signals to win on ELSA without sacrificing task performance?”

---

## 9. Methodological Pitfalls and Ethical Considerations
- **Metaphor‑driven design:** Anatomical resemblance does not guarantee computational benefit. Require ablations that show *why* a biological ingredient helps.
- **Goodharting alignment:** Optimizing a neural‑similarity metric can degrade task performance or overfit specific datasets. Use preregistered perturbation tests.
- **Cherry‑picked plausibility:** Demonstrating a mechanism on toy tasks is not evidence of scalability. Establish **external validity** across domains.
- **Opacity and overclaiming:** Claims such as “our model works like the brain” demand causal evidence, not correlational anecdotes. Mechanistic interpretability should be integral.
- **Ethics and safety:** NeuroAI may yield highly sample‑efficient, adaptive systems. Balance energy advantages with safety: robust out‑of‑distribution behavior, controllability, and societal impact should be part of the evaluation protocol.

---

## 10. A Staged Research Agenda for NeuroAI
**Stage I: Constraint‑informed components (now).**  
Isolate ingredients with biological motivation and **demonstrate independent value** under controlled comparisons: recurrence for context, multi‑timescale adaptation, modulatory gating, local learning augmenting backprop, structured sparsity for energy. Report wins and losses.

**Stage II: Joint objective models.**  
Develop agents trained under **multi‑objective** losses combining prediction, compression, control, and homeostatic regularizers. Use curricula that mirror developmental regimes (from passive observation to active manipulation). Evaluate with ELSA and causal perturbation tests.

**Stage III: Mechanism‑aligned alignment.**  
Move beyond shallow representational matches to **causal alignment**. Fit models that predict neural changes under perturbations (lesions, attention shifts, neuromodulatory manipulations). Use shared latent variables that map onto measurable biological quantities (e.g., gain, oscillatory phase, neuromodulator levels).

**Stage IV: Hardware‑constrained optimization.**  
Train models **under the constraints** of the target hardware (spiking, analog noise, quantization) rather than retrofitting after training. Co‑train with energy/latency terms and enforce locality where possible.

**Stage V: Open‑world agents.**  
Deploy embodied agents in long‑horizon, open‑world tasks with changing goals. Measure lifelong learning, safety, and cultural transmission of skills. This is where NeuroAI must ultimately prove its advantage: *durable competence under uncertainty, within real resource budgets.*

---

## 11. Conclusion
NeuroAI’s value lies not in mimicking every biological detail, nor in ignoring biology altogether, but in **constraining the design space** with principles that have stood the test of evolution: multi‑objective learning, recurrence and feedback for context, modulatory credit signals, sparse event‑driven computation, and severe resource budgets. When these constraints are cast as hypotheses and tested with the same rigor we apply to benchmarks, NeuroAI becomes an engineering science with a feedback loop to biology. The field is ready to move from metaphors and one‑off demonstrations to **reproducible, pareto‑efficient systems** that are simultaneously useful and explanatory. Achieving that will require humility about what is known, discipline in evaluation, and sustained collaboration between neuroscientists, machine learners, and hardware designers.

---

## Selected References (non‑exhaustive, for orientation)
*Note: The paper above is original prose. The following works are suggested starting points for the interested reader; exact editions and venues vary.*

- Hassabis, D., Kumaran, D., Summerfield, C., & Botvinick, M. “Neuroscience‑Inspired Artificial Intelligence.” *Neuron* (2017).
- Yamins, D. L. K., & DiCarlo, J. J. “Using goal‑driven deep learning models to understand sensory cortex.” *Nature Neuroscience* (2016).
- Lake, B. M., Ullman, T., Tenenbaum, J. B., & Gershman, S. “Building machines that learn and think like people.” *Behavioral and Brain Sciences* (2017).
- Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., & Hinton, G. “Backpropagation and the brain.” *Nature Reviews Neuroscience* (2020).
- Richards, B. A., Lillicrap, T., Beaudoin, P., et al. “A deep learning framework for neuroscience.” *Nature Neuroscience* (2019).
- Schrimpf, M., Kubilius, J., Hong, H., et al. “Brain‑Score: Which artificial neural network for object recognition is most brain‑like?” Preprint/Proceedings (2018–2020).
- Tuma, T., Pantazi, A., et al. “Stochastic phase‑change neurons.” *Nature Nanotechnology* (2016).
- Davies, M., et al. “Loihi: A Neuromorphic Manycore Processor with On‑Chip Learning.” *IEEE Micro* / arXiv (2018).
- Whittington, J. C. R., & Bogacz, R. “An approximation of backpropagation in a predictive coding network.” *Neural Computation* (2017).
- Zador, A. “A critique of pure learning and what artificial neural networks can learn from animal brains.” *Nature Communications* (2019).
