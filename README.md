# Graph+LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A collection of AWESOME things about **Graph-Related Large Language Models (LLMs)**.

Large Language Models (LLMs) have shown remarkable progress in natural language processing tasks. However, their integration with graph structures, which are prevalent in real-world applications, remains relatively unexplored. This repository aims to bridge that gap by providing a curated list of research papers that explore the intersection of graph-based techniques with LLMs.


## Table of Contents

- [Awesome-Graph-LLM ](#awesome-graph-llm-)
  - [Table of Contents](#table-of-contents)
  - [WXB Own(Alignment)](#wxb-alignment)
    - [LLM No Finetuning](#No-Finetuning)
    - [LLM Finetuning](#Finetuning)
  - [Datasets, Benchmarks \& Surveys](#datasets-benchmarks--surveys)
  - [Prompting](#prompting)
  - [General Graph Model](#general-graph-model)
  - [Large Multimodal Models (LMMs)](#large-multimodal-models-lmms)
  - [Applications](#applications)
    - [Basic Graph Reasoning](#basic-graph-reasoning)
    - [Node Classification](#node-classification)
    - [Graph Classification/Regression](#graph-classificationregression)
    - [Knowledge Graph](#knowledge-graph)
    - [Molecular Graph](#molecular-graph)
    - [Graph Robustness](#graph-robustness)
    - [Others](#others)
  - [Resources \& Tools](#resources--tools)
  - [Contributing](#contributing)
  - [Star History](#star-history)

## Wxb Alignment
### No Finetuning
- (*arXiv 2024.10*) EAMA : Entity-Aware Multimodal Alignment Based Approach for News Image Captioning [[paper](https://arxiv.org/abs/2402.19404)] [[code](NANA)]
- (*ICML 2023*) BLIP-2: Bootstrapping Language-Image Pre-training  with Frozen Image Encoders and Large Language Models [[paper](https://arxiv.org/abs/2408.14512)] [[code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)]
- (*Discover computing*) Automating the Construction of Internet Portals  with Machine Learning [[paper]([https://arxiv.org/abs/2408.14512](https://link.springer.com/article/10.1023/A:1009953814988))] [[code]

### Finetuning
- (*NeurlPS 2024*) LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings [[paper](https://arxiv.org/abs/2402.19404)] [[code](NANA)]


## Datasets, Benchmarks & Surveys
- (*NAACL'21*) Knowledge Graph Based Synthetic Corpus Generation for Knowledge-Enhanced Language Model Pre-training [[paper](https://aclanthology.org/2021.naacl-main.278/)][[code](https://github.com/google-research-datasets/KELM-corpus)]
- (*NeurIPS'23*) Can Language Models Solve Graph Problems in Natural Language? [[paper](https://arxiv.org/abs/2305.10037)][[code](https://github.com/Arthur-Heng/NLGraph)]
- (*IEEE Intelligent Systems 2023*) Integrating Graphs with Large Language Models: Methods and Prospects [[paper](https://arxiv.org/abs/2310.05499)]
- (*ICLR'24*) Talk like a Graph: Encoding Graphs for Large Language Models [[paper](https://arxiv.org/abs/2310.04560)]
- (*KDD'24*) LLM4DyG: Can Large Language Models Solve Problems on Dynamic Graphs? [[paper](https://arxiv.org/abs/2310.17110)][[code](https://github.com/wondergo2017/LLM4DyG)]
- (NeurIPS'24) TEG-DB: A Comprehensive Dataset and Benchmark of Textual-Edge Graphs [[pdf](https://arxiv.org/abs/2406.10310)][[code](https://github.com/Zhuofeng-Li/TEG-Benchmark/tree/main)][[datasets](https://huggingface.co/datasets/ZhuofengLi/TEG-Datasets/tree/main)]
- (*arXiv 2023.05*) GPT4Graph: Can Large Language Models Understand Graph Structured Data? An Empirical Evaluation and Benchmarking [[paper](https://arxiv.org/abs/2305.15066)][[code](https://github.com/SpaceLearner/Graph-GPT)]
- (*arXiv 2023.08*) Graph Meets LLMs: Towards Large Graph Models [[paper](http://arxiv.org/abs/2308.14522)]
- (*arXiv 2023.10*) Towards Graph Foundation Models: A Survey and Beyond [[paper](https://arxiv.org/abs/2310.11829v1)]
- (*arXiv 2023.11*) Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey [[paper](https://arxiv.org/abs/2311.07914v1)]
- (*arXiv 2023.11*) A Survey of Graph Meets Large Language Model: Progress and Future Directions [[paper](https://arxiv.org/abs/2311.12399)][[code](https://github.com/yhLeeee/Awesome-LLMs-in-Graph-tasks)]
- (*arXiv 2023.12*) Large Language Models on Graphs: A Comprehensive Survey [[paper](https://arxiv.org/abs/2312.02783)][[code](https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs)]
- (*arXiv 2024.02*) Towards Versatile Graph Learning Approach: from the Perspective of Large Language Models [[paper](https://arxiv.org/abs/2402.11641)]
- (*arXiv 2024.04*) Graph Machine Learning in the Era of Large Language Models (LLMs) [[paper](https://arxiv.org/abs/2404.14928)]
- (*arXiv 2024.05*) A Survey of Large Language Models for Graphs [[paper](https://arxiv.org/abs/2405.08011)][[code](https://github.com/HKUDS/Awesome-LLM4Graph-Papers)]
- (*NeurIPS'24 D&B*) GLBench: A Comprehensive Benchmark for Graph with Large Language Models [[paper](https://arxiv.org/abs/2407.07457)][[code](https://github.com/NineAbyss/GLBench)]
- (*arXiv 2024.07*) Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness [[paper](https://arxiv.org/abs/2407.12068)][[code](https://github.com/KaiGuo20/GraphLLM_Robustness)]
- (*Complex Networks 2024*) LLMs hallucinate graphs too: a structural perspective [[paper](https://arxiv.org/abs/2409.00159)]
- (*arXiv 2024.10*) Can Graph Descriptive Order Affect Solving Graph Problems with LLMs? [[paper](https://arxiv.org/abs/2402.07140)]
- (*arXiv 2024.10*) How Do Large Language Models Understand Graph Patterns? A Benchmark for Graph Pattern Comprehension [[paper](https://arxiv.org/abs/2410.05298v1)]
- (*arXiv 2024.10*) GRS-QA - Graph Reasoning-Structured Question Answering Dataset [[paper](https://arxiv.org/abs/2411.00369)]
- (*NeurIPS'24 D&B*) Can Large Language Models Analyze Graphs like Professionals? A Benchmark, Datasets and Models [[paper](https://arxiv.org/abs/2409.19667)] [[code](https://github.com/BUPT-GAMMA/ProGraph)] 
  
## Prompting
- (*EMNLP'23*) StructGPT: A General Framework for Large Language Model to Reason over Structured Data [[paper](https://arxiv.org/abs/2305.09645)][[code](https://github.com/RUCAIBox/StructGPT)]
- (*AAAI'24*) Graph of Thoughts: Solving Elaborate Problems with Large Language Models [[paper](https://arxiv.org/abs/2308.09687)][[code](https://github.com/spcl/graph-of-thoughts)]
- (*arXiv 2023.05*) PiVe: Prompting with Iterative Verification Improving Graph-based Generative Capability of LLMs [[paper](https://arxiv.org/abs/2305.12392)][[code](https://github.com/Jiuzhouh/PiVe)]
- (*arXiv 2023.08*) Boosting Logical Reasoning in Large Language Models through a New Framework: The Graph of Thought [[paper](https://arxiv.org/abs/2308.08614)]
- (*arxiv 2023.10*) Thought Propagation: An Analogical Approach to Complex Reasoning with Large Language Models [[paper](https://arxiv.org/abs/2310.03965v2)]
- (*arxiv 2024.01*) Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts [[paper](https://arxiv.org/abs/2401.14295)]
- (*ACL'24*) Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs [[paper](https://arxiv.org/abs/2404.07103)][[code](https://github.com/PeterGriffinJin/Graph-CoT)]


## General Graph Model
- (*ICLR'24*) One for All: Towards Training One Graph Model for All Classification Tasks [[paper](https://arxiv.org/abs/2310.00149)][[code](https://github.com/LechengKong/OneForAll)]
- (WWW'24) GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks [[paper](https://arxiv.org/abs/2402.07197)][[code](https://github.com/alibaba/GraphTranslator?tab=readme-ov-file)]
- (*arXiv 2023.08*) Natural Language is All a Graph Needs [[paper](https://arxiv.org/abs/2308.07134)][[code](https://github.com/agiresearch/InstructGLM)]
- (*arXiv 2023.10*) GraphGPT: Graph Instruction Tuning for Large Language Models [[paper](https://arxiv.org/abs/2310.13023)][[code](https://github.com/HKUDS/GraphGPT)][[blog in Chinese](https://mp.weixin.qq.com/s/rvKTFdCk719Q6hT09Caglw)]
- (*arXiv 2023.10*) Graph Agent: Explicit Reasoning Agent for Graphs [[paper](https://arxiv.org/abs/2310.16421)]
- (*arXiv 2024.02*) Let Your Graph Do the Talking: Encoding Structured Data for LLMs [[paper](https://arxiv.org/abs/2402.05862)]
- (*NeurIPS'24*) G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering [[paper](https://arxiv.org/abs/2402.07630)][[code](https://github.com/XiaoxinHe/G-Retriever)][[blog](https://medium.com/@xxhe/graph-retrieval-augmented-generation-rag-beb19dc30424)]
- (*arXiv 2024.02*) InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment [[paper](https://arxiv.org/abs/2402.08785)][[code](https://github.com/wjn1996/InstructGraph)]
- (*arXiv 2024.02*) LLaGA: Large Language and Graph Assistant [[paper](https://arxiv.org/abs/2402.08170)][[code](https://github.com/VITA-Group/LLaGA)]
- (*arXiv 2024.02*) HiGPT: Heterogeneous Graph Language Model [[paper](https://arxiv.org/abs/2402.16024)][[code](https://github.com/HKUDS/HiGPT)]
- (*arXiv 2024.02*) UniGraph: Learning a Cross-Domain Graph Foundation Model From Natural Language [[paper](https://arxiv.org/abs/2402.13630)]
- (*arXiv 2024.06*) UniGLM: Training One Unified Language Model for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2406.12052)][[code](https://github.com/NYUSHCS/UniGLM)]
- (*arXiv 2024.07*) GOFA: A Generative One-For-All Model for Joint Graph Language Modeling [[paper](https://arxiv.org/abs/2407.09709)][[code](https://github.com/JiaruiFeng/GOFA)]
- (*arXiv 2024.08*) AnyGraph: Graph Foundation Model in the Wild [[paper](https://arxiv.org/abs/2408.10700)][[code](https://github.com/HKUDS/AnyGraph)]
- (*arXiv 2024.10*) NT-LLM: A Novel Node Tokenizer for Integrating Graph Structure into Large Language Models [[paper](https://arxiv.org/abs/2410.10743)]


## Large Multimodal Models (LMMs)
- (*NeurIPS'23*) GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph [[paper](https://arxiv.org/abs/2309.13625)][[code](https://github.com/lixinustc/GraphAdapter)]
- (*arXiv 2023.10*) Multimodal Graph Learning for Generative Tasks [[paper](https://arxiv.org/abs/2310.07478)][[code](https://github.com/minjiyoon/MMGL)]
- (*arXiv 2024.02*) Rendering Graphs for Graph Reasoning in Multimodal Large Language Models [[paper](https://arxiv.org/abs/2402.02130)]
- (*ACL 2024*) Graph Language Models [[paper](https://aclanthology.org/2024.acl-long.245/)][[code](https://github.com/Heidelberg-NLP/GraphLanguageModels)]
- (*NeurIPS'24*) GITA: Graph to Visual and Textual Integration for Vision-Language Graph Reasoning [[paper](https://arxiv.org/abs/2402.02130)][[code](https://github.com/WEIYanbin1999/GITA)][[project](https://v-graph.github.io/)]

## Applications
### Basic Graph Reasoning
- (*KDD'24*) GraphWiz: An Instruction-Following Language Model for Graph Problems [[paper](https://arxiv.org/abs/2402.16029)][[code](https://github.com/nuochenpku/Graph-Reasoning-LLM)][[project](https://graph-wiz.github.io/)]
- (*arXiv 2023.04*) Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT [[paper](https://arxiv.org/abs/2304.11116)][[code](https://github.com/jwzhanggy/Graph_Toolformer)]
- (*arXiv 2023.10*) GraphText: Graph Reasoning in Text Space [[paper](https://arxiv.org/abs/2310.01089)]
- (*arXiv 2023.10*) GraphLLM: Boosting Graph Reasoning Ability of Large Language Model [[paper](https://arxiv.org/abs/2310.05845)][[code](https://github.com/mistyreed63849/Graph-LLM)]
- (*arXiv 2024.10*) GUNDAM: Aligning Large Language Models with Graph Understanding [[paper](https://arxiv.org/abs/2410.01457)]
- (*arXiv 2024.10*) Are Large-Language Models Graph Algorithmic Reasoners? [[paper](https://arxiv.org/abs/2410.22597)][[code](https://github.com/ataylor24/MAGMA)]
- (*arXiv 2024.10*) GCoder: Improving Large Language Model for Generalized Graph Problem Solving [[paper](https://arxiv.org/pdf/2410.19084)] [[code](https://github.com/Bklight999/WWW25-GCoder/tree/master)]
- (*arXiv 2024.10*) GraphTeam: Facilitating Large Language Model-based Graph Analysis via Multi-Agent Collaboration [[paper](https://arxiv.org/abs/2410.18032)] [[code](https://github.com/BUPT-GAMMA/GraphTeam)]

### Node Classification
- (*ICLR'24*) Explanations as Features: LLM-Based Features for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2305.19523)][[code](https://github.com/XiaoxinHe/TAPE)]
- (*ICLR'24*) Label-free Node Classification on Graphs with Large Language Models (LLMS) [[paper](https://arxiv.org/abs/2310.04668)]
- (*WWW'24*) Can GNN be Good Adapter for LLMs? [[paper](https://arxiv.org/html/2402.12984v1)][[code](https://github.com/zjunet/GraphAdapter)]
- (*CIKM'24*) Distilling Large Language Models for Text-Attributed Graph Learning [[paper](https://arxiv.org/abs/2402.12022)]
- (*arXiv 2023.07*) Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs [[paper](https://arxiv.org/abs/2307.03393)][[code](https://github.com/CurryTang/Graph-LLM)]
- (*arXiv 2023.09*) Can LLMs Effectively Leverage Structural Information for Graph Learning: When and Why [[paper](https://arxiv.org/abs/2309.16595)][[code](https://github.com/TRAIS-Lab/LLM-Structured-Data)]
- (*arXiv 2023.10*) Empower Text-Attributed Graphs Learning with Large Language Models (LLMs) [[paper](https://arxiv.org/abs/2310.09872)]
- (*arXiv 2023.10*) Disentangled Representation Learning with Large Language Models for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2310.18152)]
- (*arXiv 2023.11*) Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs [[paper](https://arxiv.org/abs/2311.14324)]
- (*arXiv 2024.01*) Efficient Tuning and Inference for Large Language Models on Textual Graphs [[paper](https://arxiv.org/abs/2401.15569)][[code](https://github.com/ZhuYun97/ENGINE)]
- (*arXiv 2024.02*) Similarity-based Neighbor Selection for Graph LLMs [[paper](https://arxiv.org/abs/2402.03720)] [[code](https://github.com/ruili33/SNS)]
- (*arXiv 2024.02*) Distilling Large Language Models for Text-Attributed Graph Learning [[paper](https://arxiv.org/abs/2402.12022)]
- (*arXiv 2024.02*) GraphEdit: Large Language Models for Graph Structure Learning [[paper](https://arxiv.org/abs/2402.15183)][[code](https://github.com/HKUDS/GraphEdit?tab=readme-ov-file)]
- (*arXiv 2024.05*) LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework [[paper](https://arxiv.org/abs/2405.13902)][[code](https://github.com/QiaoYRan/LOGIN)]
- (*arXiv 2024.06*) GAugLLM: Improving Graph Contrastive Learning for Text-Attributed Graphs with Large Language Models [[paper](https://arxiv.org/abs/2406.11945)][[code](https://github.com/NYUSHCS/GAugLLM)]
- (*arXiv 2024.07*) Enhancing Data-Limited Graph Neural Networks by Actively Distilling Knowledge from Large Language Models [[paper](https://arxiv.org/abs/2407.13989)]
- (*arXiv 2024.07*) All Against Some: Efficient Integration of Large Language Models for Message Passing in Graph Neural Networks [[paper](https://arxiv.org/abs/2407.14996)]
- (*arXiv 2024.10*) Let's Ask GNN: Empowering Large Language Model for Graph In-Context Learning [[paper](https://arxiv.org/abs/2410.07074)]
- (*arXiv 2024.10*) Large Language Model-based Augmentation for Imbalanced Node Classification on Text-Attributed Graphs [[paper](https://arxiv.org/abs/2410.16882)]
- (*arXiv 2024.10*) Enhance Graph Alignment for Large Language Models [[paper](https://arxiv.org/abs/2410.11370)]

### Graph Classification/Regression
- (*arXiv 2023.06*) GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning [[paper](https://arxiv.org/abs/2306.13089)][[code](https://github.com/zhao-ht/GIMLET)]
- (*arXiv 2023.07*) Can Large Language Models Empower Molecular Property Prediction? [[paper](https://arxiv.org/abs/2307.07443)][[code](https://github.com/ChnQ/LLM4Mol)]


### Knowledge Graph
- (*AAAI'22*) Enhanced Story Comprehension for Large Language Models through Dynamic Document-Based Knowledge Graphs [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/21286)]
- (*EMNLP'22*) Language Models of Code are Few-Shot Commonsense Learners [[paper](https://arxiv.org/abs/2210.07128)][[code](https://github.com/reasoning-machines/CoCoGen)]
- (*SIGIR'23*) Schema-aware Reference as Prompt Improves Data-Efficient Knowledge Graph Construction [[paper](https://arxiv.org/abs/2210.10709)][[code](https://github.com/zjunlp/RAP)]
- (*TKDE‘23*) AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models [[paper](https://arxiv.org/abs/2307.11772)][[code](https://github.com/ruizhang-ai/AutoAlign)]
- (*AAAI'24*) Graph Neural Prompting with Large Language Models [[paper](https://arxiv.org/abs/2309.15427)][[code](https://github.com/meettyj/GNP)]
- (*NAACL'24*) zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models [[paper](https://arxiv.org/abs/2311.10112)]
- (*ICLR'24*) Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph [[paper](https://arxiv.org/abs/2307.07697)][[code](https://github.com/IDEA-FinAI/ToG)]
- (*arXiv 2023.04*) CodeKGC: Code Language Model for Generative Knowledge Graph Construction [[paper](https://arxiv.org/abs/2304.09048)][[code](https://github.com/zjunlp/DeepKE/tree/main/example/llm/CodeKGC)]
- (*arXiv 2023.05*) Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs [[paper](https://arxiv.org/abs/2305.09858)]
- (*arXiv 2023.08*) MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models [[paper](https://arxiv.org/abs/2308.09729)][[code](https://github.com/wyl-willing/MindMap)]
- (*arXiv 2023.10*) Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph [[paper](https://arxiv.org/abs/2310.16452)]
- (*arXiv 2023.10*) Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning [[paper](https://arxiv.org/abs/2310.01061)][[code](https://github.com/RManLuo/reasoning-on-graphs)]
- (*arXiv 2023.11*) Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models [[paper](https://arxiv.org/abs/2311.10112)]
- (*arXiv 2023.12*) KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn’t Know [[paper](https://arxiv.org/abs/2312.11539)]
- (*arXiv 2024.02*) Large Language Model Meets Graph Neural Network in Knowledge Distillation [[paper](https://arxiv.org/abs/2402.05894)]
- (*arXiv 2024.02*) Large Language Models Can Learn Temporal Reasoning [[paper](https://arxiv.org/pdf/2401.06853v2.pdf)][[code](https://github.com/xiongsiheng/TG-LLM)]
- (*arXiv 2024.02*) Knowledge Graph Large Language Model (KG-LLM) for Link Prediction [[paper](https://arxiv.org/abs/2403.07311)]
- (*arXiv 2024.03*) Call Me When Necessary: LLMs can Efficiently and Faithfully Reason over Structured Environments [[paper](https://arxiv.org/abs/2403.08593)]
- (*arXiv 2024.04*) Evaluating the Factuality of Large Language Models using Large-Scale Knowledge Graphs [[paper](https://arxiv.org/abs/2404.00942)][[code](https://github.com/xz-liu/GraphEval)]
- (*arXiv 2024.04*) Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction [[paper](https://arxiv.org/abs/2404.03868)][[code](https://github.com/clear-nus/edc)]
- (*arXiv 2024.05*) FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering [[paper](https://arxiv.org/abs/2405.13873)]
- (*arXiv 2024.06*) Explore then Determine: A GNN-LLM Synergy Framework for Reasoning over Knowledge Graph [[paper](https://arxiv.org/abs/2406.01145)]
- (*ACL 2024*) Graph Language Models [[paper](https://aclanthology.org/2024.acl-long.245/)][[code](https://github.com/Heidelberg-NLP/GraphLanguageModels)]
- (*EMNLP 2024*) LLM-Based Multi-Hop Question Answering with Knowledge Graph Integration in Evolving Environments [[paper]](https://arxiv.org/abs/2408.15903)

### Molecular Graph
- (*arXiv 2024.06*) MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction [[paper](https://arxiv.org/abs/2406.12950)][[code](https://github.com/NYUSHCS/MolecularGPT)]
- (*arXiv 2024.06*) HIGHT: Hierarchical Graph Tokenization for Graph-Language Alignment [[paper](https://arxiv.org/abs/2406.14021)][[project](https://higraphllm.github.io/)]
- (*arXiv 2024.06*) MolX: Enhancing Large Language Models for Molecular Learning with A Multi-Modal Extension [[paper](https://arxiv.org/abs/2406.06777)]
- (*arXiv 2024.06*) LLM and GNN are Complementary: Distilling LLM for Multimodal Graph Learning [[paper](https://arxiv.org/abs/2406.01032)]
- (*arXiv 2024.10*) G2T-LLM: Graph-to-Tree Text Encoding for Molecule Generation with Fine-Tuned Large Language Models [[paper](https://arxiv.org/abs/2410.02198v1)]

### Graph Robustness
- (*arXiv 2024.05*) Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level [[paper](https://arxiv.org/abs/2405.16405)]
- (*arXiv 2024.08*) Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks? [[paper](https://arxiv.org/pdf/2408.08685)]

### Others
- (*WSDM'24*) LLMRec: Large Language Models with Graph Augmentation for Recommendation [[paper](https://arxiv.org/abs/2311.00423)][[code](https://github.com/HKUDS/LLMRec)][[blog in Chinese](https://mp.weixin.qq.com/s/aU-uzLWH6xfIuoon-Zq8Cg)].
- (*arXiv 2023.03*) Ask and You Shall Receive (a Graph Drawing): Testing ChatGPT’s Potential to Apply Graph Layout Algorithms [[paper](https://arxiv.org/abs/2303.08819)]
- (*arXiv 2023.05*) Graph Meets LLM: A Novel Approach to Collaborative Filtering for Robust Conversational Understanding [[paper](https://arxiv.org/abs/2305.14449)]
- (*arXiv 2023.05*) ChatGPT Informed Graph Neural Network for Stock Movement Prediction [[paper](https://arxiv.org/abs/2306.03763)][[code](https://github.com/ZihanChen1995/ChatGPT-GNN-StockPredict)]
- (*arXiv 2023.10*) Graph Neural Architecture Search with GPT-4 [[paper](https://arxiv.org/abs/2310.01436)]
- (*arXiv 2023.11*) Biomedical knowledge graph-enhanced prompt generation for large language models [[paper](https://arxiv.org/abs/2311.17330)][[code](https://github.com/BaranziniLab/KG_RAG)]
- (*arXiv 2023.11*) Graph-Guided Reasoning for Multi-Hop Question Answering in Large Language Models [[paper](https://arxiv.org/abs/2311.09762)]
- (*NeurIPS'24*) Microstructures and Accuracy of Graph Recall by Large Language Models [[paper](https://arxiv.org/abs/2402.11821)][[code](https://github.com/Abel0828/llm-graph-recall)]
- (*arXiv 2024.02*) Causal Graph Discovery with Retrieval-Augmented Generation based Large Language Models [[paper](https://arxiv.org/abs/2402.15301)]
- (*arXiv 2024.02*) Graph-enhanced Large Language Models in Asynchronous Plan Reasoning [[paper](https://arxiv.org/abs/2402.02805)][[code](https://github.com/fangru-lin/graph-llm-asynchow-plan)]
- (*arXiv 2024.02*) Efficient Causal Graph Discovery Using Large Language Models [[paper](https://arxiv.org/abs/2402.01207)]
- (*arXiv 2024.03*) Exploring the Potential of Large Language Models in Graph Generation [[paper](https://arxiv.org/abs/2403.14358)]
- (*arXiv 2024.05*) Don't Forget to Connect! Improving RAG with Graph-based Reranking [[paper](https://arxiv.org/abs/2405.18414)]
- (*NeurIPS'24*) Can Graph Learning Improve Planning in LLM-based Agents? [[paper](https://arxiv.org/abs/2405.19119)][[code](https://github.com/WxxShirley/GNN4TaskPlan)]
- (*arXiv 2024.06*) GNN-RAG: Graph Neural Retrieval for Large Language Modeling Reasoning [[paper](https://arxiv.org/abs/2405.20139)][[code](https://github.com/cmavro/GNN-RAG)]
- (*arXiv 2024.07*) LLMExplainer: Large Language Model based Bayesian Inference for Graph Explanation Generation [[paper](https://arxiv.org/abs/2407.15351)]
- (*arXiv 2024.08*) CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases [[paper](https://arxiv.org/abs/2408.03910)][[code](https://github.com/modelscope/modelscope-agent/tree/master/apps/codexgraph_agent)][[project](https://laptype.github.io/CodexGraph-page/)]
- (*arXiv 2024.10*) Graph Linearization Methods for Reasoning on Graphs with Large Language Models [[paper](https://arxiv.org/abs/2410.19494)]
- (*arXiv 2024.10*) GraphRouter: A Graph-based Router for LLM Selections [[paper](https://arxiv.org/abs/2410.03834)][[code](https://github.com/ulab-uiuc/GraphRouter)]
- (*arXiv 2024.10*) Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs [[paper](https://arxiv.org/abs/2410.11001)] [[code](https://github.com/ulab-uiuc/GoR)]
- (*arXiv 2024.10*) G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks [[paper](https://arxiv.org/abs/2410.11782)] [[code](https://anonymous.4open.science/r/GDesigner-3063)]



## Resources & Tools
- [GraphGPT: Extrapolating knowledge graphs from unstructured text using GPT-3](https://github.com/varunshenoy/GraphGPT)
- [GraphML: Graph markup language](https://cs.brown.edu/people/rtamassi/gdhandbook/chapters/graphml.pdf). An XML-based file format for graphs.
- [GML: Graph modelling language](https://networkx.org/documentation/stable/reference/readwrite/gml.html). Read graphs in GML format.

## Contributing
👍 Contributions to this repository are welcome! 

If you have come across relevant resources, feel free to open an issue or submit a pull request.
```
- (*conference|journal*) paper_name [[pdf](link)][[code](link)]
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XiaoxinHe/Awesome-Graph-LLM&type=Date)](https://star-history.com/#XiaoxinHe/Awesome-Graph-LLM&Date)

==========================================survey==========================================================================================================
<h1 align="center"> Awesome-LLMs-in-Graph-tasks </a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg) ![GitHub stars](https://img.shields.io/github/stars/yhLeeee/Awesome-LLMs-in-Graph-tasks.svg)

</h5>

> This is a collection of papers on leveraging **Large Language Models** in **Graph Tasks**. 
It's based on our survey paper: [A Survey of Graph Meets Large Language Model: Progress and Future Directions](https://arxiv.org/abs/2311.12399). 

> We will try to make this list updated frequently. If you found any error or any missed paper, please don't hesitate to open issues or pull requests.

> Our survey has been accepted by IJCAI 2024 survey track.

## How can LLMs help improve graph-related tasks?

With the help of LLMs, there has been a notable shift in the way we interact with graphs, particularly those containing nodes associated with text attributes. The integration of LLMs with traditional GNNs can be mutually beneficial and enhance graph learning. While GNNs are proficient at capturing structural information, they primarily rely on semantically constrained embeddings as node features, limiting their ability to express the full complexities of the nodes. Incorporating LLMs, GNNs can be enhanced with stronger node features that effectively capture both structural and contextual aspects. On the other hand, LLMs excel at encoding text but often struggle to capture structural information present in graph data. Combining GNNs with LLMs can leverage the robust textual understanding of LLMs while harnessing GNNs' ability to capture structural relationships, leading to more comprehensive and powerful graph learning.

<p align="center"><img src="Figures/overview.png" width=75% height=75%></p>
<p align="center"><em>Figure 1.</em> The overview of Graph Meets LLMs.</p>


## Summarizations based on proposed taxonomy

<p align="center"><img src="Figures/summarization.png" width=100% height=75%></p>

<p align="left"><em>Table 1.</em> A summary of models that leverage LLMs to assist graph-related tasks in literature, ordered by their release time. <b>Fine-tuning</b> denotes whether it is necessary to fine-tune the parameters of LLMs, and &hearts; indicates that models employ parameter-efficient fine-tuning (PEFT) strategies, such as LoRA and prefix tuning. <b>Prompting</b> indicates the use of text-formatted prompts in LLMs, done manually or automatically. Acronyms in <b>Task</b>: Node refers to node-level tasks; Link refers to link-level tasks; Graph refers to graph-level tasks; Reasoning refers to Graph Reasoning; Retrieval refers to Graph-Text Retrieval; Captioning refers to Graph Captioning.</p >

## Table of Contents

- [Awesome-LLMs-in-Graph-tasks](#awesome-llms-in-graph-tasks)
  - [How can LLMs help improve graph-related tasks](#how-can-llms-help-improve-graph-related-tasks)
  - [Summarizations based on proposed taxonomy](#summarizations-based-on-proposed-taxonomy)
  - [Table of Contents](#table-of-contents)
  - [LLM as Enhancer](#llm-as-enhancer)
  - [LLM as Predictor](#llm-as-predictor)
  - [GNN-LLM Alignment](#gnn-llm-alignment)
  - [Others](#others)
  - [Contributing](#contributing)
  - [Cite Us](#cite-us)


## LLM as Enhancer
* (_2022.03_) [ICLR' 2022] **Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction** [[Paper](https://arxiv.org/abs/2111.00064) | [Code](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt)]
   <details close>
   <summary>GIANT</summary>
   <p align="center"><img width="75%" src="Figures/GIANT.jpg" /></p>
   <p align="center"><em>The framework of GIANT.</em></p>
   </details>
* (_2023.02_) [ICLR' 2023] **Edgeformers: Graph-Empowered Transformers for Representation Learning on Textual-Edge Networks** [[Paper](https://arxiv.org/abs/2302.11050) | [Code](https://github.com/PeterGriffinJin/Edgeformers)]
   <details close>
   <summary>Edgeformers</summary>
   <p align="center"><img width="75%" src="Figures/Edgeformers.jpg" /></p>
   <p align="center"><em>The framework of Edgeformers.</em></p>
   </details>
* (_2023.05_) [KDD' 2023] **Graph-Aware Language Model Pre-Training on a Large Graph Corpus Can Help Multiple Graph Applications** [[Paper](https://arxiv.org/abs/2306.02592)]
   <details close>
   <summary>GALM</summary>
   <p align="center"><img width="75%" src="Figures/GALM.jpg" /></p>
   <p align="center"><em>The framework of GALM.</em></p>
   </details>
* (_2023.06_) [KDD' 2023] **Heterformer: Transformer-based Deep Node Representation Learning on Heterogeneous Text-Rich Networks** [[Paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599376?casa_token=M9bG1HLyTEYAAAAA:gIiYO9atgtxNaBgfKpy4D3N66QDkCFLFvlEADvzC8Pobe_EWausOknGnRFzdDF-Xnq-vbWAWMT1qkA) | [Code](https://github.com/PeterGriffinJin/Heterformer)]
   <details close>
   <summary>Heterformer</summary>
   <p align="center"><img width="75%" src="Figures/Heterformers.jpg" /></p>
   <p align="center"><em>The framework of Heterformer.</em></p>
   </details>
* (_2023.05_) [ICLR' 2024] **Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning** [[Paper](https://arxiv.org/abs/2305.19523) | [Code](https://github.com/XiaoxinHe/TAPE)]
   <details close>
   <summary>TAPE</summary>
   <p align="center"><img width="75%" src="Figures/TAPE.jpg" /></p>
   <p align="center"><em>The framework of TAPE.</em></p>
   </details>
* (_2023.08_) [Arxiv' 2023] **Exploring the potential of large language models (llms) in learning on graphs** [[Paper](https://arxiv.org/abs/2307.03393)]
   <details close>
   <summary>KEA</summary>
   <p align="center"><img width="75%" src="Figures/KEA.jpg" /></p>
   <p align="center"><em>The framework of KEA.</em></p>
   </details>
* (_2023.07_) [Arxiv' 2023] **Can Large Language Models Empower Molecular Property Prediction?** [[Paper](https://arxiv.org/abs/2307.07443) | [Code](https://github.com/ChnQ/LLM4Mol)]
   <details close>
   <summary>LLM4Mol</summary>
   <p align="center"><img width="75%" src="Figures/LLM4Mol.jpg" /></p>
   <p align="center"><em>The framework of LLM4Mol.</em></p>
   </details>
* (_2023.08_) [Arxiv' 2023] **Simteg: A frustratingly simple approach improves textual graph learning** [[Paper](https://arxiv.org/abs/2308.02565) | [Code](https://github.com/vermouthdky/SimTeG)]
   <details close>
   <summary>SimTeG</summary>
   <p align="center"><img width="75%" src="Figures/SimTeG.jpg" /></p>
   <p align="center"><em>The framework of SimTeG.</em></p>
   </details>
* (_2023.09_) [Arxiv' 2023] **Prompt-based Node Feature Extractor for Few-shot Learning on Text-Attributed Graphs** [[Paper](https://arxiv.org/abs/2309.02848)]
   <details close>
   <summary>G-Prompt</summary>
   <p align="center"><img width="75%" src="Figures/G-Prompt.jpg" /></p>
   <p align="center"><em>The framework of G-Prompt.</em></p>
   </details>
* (_2023.09_) [Arxiv' 2023] **TouchUp-G: Improving Feature Representation through Graph-Centric Finetuning** [[Paper](https://arxiv.org/abs/2309.13885)]
   <details close>
   <summary>TouchUp-G</summary>
   <p align="center"><img width="75%" src="Figures/TouchUp-G.jpg" /></p>
   <p align="center"><em>The framework of TouchUp-G.</em></p>
   </details>
* (_2023.09_) [ICLR' 2024] **One for All: Towards Training One Graph Model for All Classification Tasks** [[Paper](https://arxiv.org/abs/2310.00149) | [Code](https://github.com/LechengKong/OneForAll)]
   <details close>
   <summary>OFA</summary>
   <p align="center"><img width="75%" src="Figures/OFA.jpg" /></p>
   <p align="center"><em>The framework of OFA.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **Learning Multiplex Embeddings on Text-rich Networks with One Text Encoder** [[Paper](https://arxiv.org/abs/2310.06684) | [Code](https://github.com/PeterGriffinJin/METERN-submit)]
   <details close>
   <summary>METERN</summary>
   <p align="center"><img width="75%" src="Figures/METERN.jpg" /></p>
   <p align="center"><em>The framework of METERN.</em></p>
   </details>
* (_2023.11_) [WSDM' 2024] **LLMRec: Large Language Models with Graph Augmentation for Recommendation** [[Paper](https://arxiv.org/abs/2311.00423) | [Code](https://github.com/HKUDS/LLMRec)]
   <details close>
   <summary>LLMRec</summary>
   <p align="center"><img width="75%" src="Figures/LLMRec.jpg" /></p>
   <p align="center"><em>The framework of LLMRec.</em></p>
   </details>
* (_2023.11_) [NeurIPS' 2023] **WalkLM: A Uniform Language Model Fine-tuning Framework for Attributed Graph Embedding** [[Paper](https://openreview.net/forum?id=ZrG8kTbt70) | [Code](https://github.com/Melinda315/WalkLM)]
   <details close>
   <summary>WalkLM</summary>
   <p align="center"><img width="75%" src="Figures/WalkLM.jpg" /></p>
   <p align="center"><em>The framework of WalkLM.</em></p>
   </details>
* (_2024.01_) [IJCAI' 2024] **Efficient Tuning and Inference for Large Language Models on Textual Graphs** [[Paper](https://arxiv.org/abs/2401.15569)]
   <details close>
   <summary>ENGINE</summary>
   <p align="center"><img width="75%" src="Figures/ENGINE.jpg" /></p>
   <p align="center"><em>The framework of ENGINE.</em></p>
   </details>
* (_2024.02_) [KDD' 2024] **ZeroG: Investigating Cross-dataset Zero-shot Transferability in Graphs** [[Paper](https://arxiv.org/abs/2402.11235)]
   <details close>
   <summary>ZeroG</summary>
   <p align="center"><img width="75%" src="Figures/ZeroG.jpg" /></p>
   <p align="center"><em>The framework of ZeroG.</em></p>
   </details>
* (_2024.02_) [Arxiv' 2024] **UniGraph: Learning a Cross-Domain Graph Foundation Model From Natural Language** [[Paper](https://arxiv.org/abs/2402.13630)]
   <details close>
   <summary>UniGraph</summary>
   <p align="center"><img width="75%" src="Figures/UniGraph.jpg" /></p>
   <p align="center"><em>The framework of UniGraph.</em></p>
   </details>

* (_2024.02_) [CIKM' 2024] **Distilling Large Language Models for Text-Attributed Graph Learning** [[Paper](https://arxiv.org/abs/2402.12022)]
   <details close>
   <summary>Pan, et al.</summary>
   <p align="center"><img width="75%" src="Figures/Pan_etal.jpg" /></p>
   <p align="center"><em>The framework of Pan, et al.</em></p>
   </details>
   
* (_2024.10_) [CIKM' 2024] **When LLM Meets Hypergraph: A Sociological Analysis on Personality via Online Social Networks** [[Paper](https://arxiv.org/abs/2407.03568) | [Code](https://github.com/ZhiyaoShu/LLM-HGNN-MBTI)]
   <details close>
   <summary>Shu, et al.</summary>
   <p align="center"><img width="75%" src="Figures/shu2024llm.jpg" /></p>
   <p align="center"><em>The framework of Shu, et al.</em></p>
   </details>
   

## LLM as Predictor

* (_2023.05_) [NeurIPS' 2023] **Can language models solve graph problems in natural language?** [[Paper](https://arxiv.org/abs/2305.10037) | [Code](https://github.com/Arthur-Heng/NLGraph)]
   <details close>
   <summary>NLGraph</summary>
   <p align="center"><img width="75%" src="Figures/NLGraph.jpg" /></p>
   <p align="center"><em>The framework of NLGraph.</em></p>
   </details>
* (_2023.05_) [Arxiv' 2023] **GPT4Graph: Can Large Language Models Understand Graph Structured Data? An Empirical Evaluation and Benchmarking** [[Paper](https://arxiv.org/abs/2305.15066) | [Code](https://anonymous.4open.science/r/GPT4Graph)]
   <details close>
   <summary>GPT4Graph</summary>
   <p align="center"><img width="75%" src="Figures/GPT4Graph.png" /></p>
   <p align="center"><em>The framework of GPT4Graph.</em></p>
   </details>
* (_2023.06_) [NeurIPS' 2023] **GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning** [[Paper](https://arxiv.org/abs/2306.13089) | [Code](https://github.com/zhao-ht/GIMLET)]
   <details close>
   <summary>GIMLET</summary>
   <p align="center"><img width="75%" src="Figures/GIMLET.jpg" /></p>
   <p align="center"><em>The framework of GIMLET.</em></p>
   </details>
* (_2023.07_) [Arxiv' 2023] **Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs** [[Paper](https://arxiv.org/abs/2307.03393) | [Code](https://github.com/CurryTang/Graph-LLM)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="Figures/Chen et al.jpg" /></p>
   <p align="center"><em>The designed prompts of Chen et al.</em></p>
   </details>
* (_2023.08_) [Arxiv' 2023] **GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text** [[Paper](https://arxiv.org/abs/2308.06911)]
   <details close>
   <summary>GIT-Mol</summary>
   <p align="center"><img width="75%" src="Figures/GIT-Mol.jpg" /></p>
   <p align="center"><em>The framework of GIT-Mol.</em></p>
   </details>
* (_2023.08_) [Arxiv' 2023] **Natural Language is All a Graph Needs** [[Paper](http://arxiv.org/abs/2308.07134) | [Code](https://github.com/agiresearch/InstructGLM)]
   <details close>
   <summary>InstructGLM</summary>
   <p align="center"><img width="75%" src="Figures/InstructGLM.jpg" /></p>
   <p align="center"><em>The framework of InstructGLM.</em></p>
   </details>
* (_2023.08_) [Arxiv' 2023] **Evaluating Large Language Models on Graphs: Performance Insights and Comparative Analysis** [[Paper](https://arxiv.org/abs/2308.11224) | [Code](https://github.com/Ayame1006/LLMtoGraph)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="Figures/Liu et al.jpg" /></p>
   <p align="center"><em>The designed prompts of Liu et al.</em></p>
   </details>
* (_2023.09_) [Arxiv' 2023] **Can LLMs Effectively Leverage Graph Structural Information: When and Why** [[Paper](https://arxiv.org/abs/2309.16595) | [Code](https://github.com/TRAIS-Lab/LLM-Structured-Data)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="Figures/Huang et al.jpg" /></p>
   <p align="center"><em>The designed prompts of Huang et al.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **GraphText: Graph Reasoning in Text Space** [[Paper](https://arxiv.org/abs/2310.01089)] | [Code](https://github.com/AndyJZhao/GraphText)]
   <details close>
   <summary>GraphText</summary>
   <p align="center"><img width="75%" src="Figures/GraphText.jpg" /></p>
   <p align="center"><em>The framework of GraphText.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **Talk like a Graph: Encoding Graphs for Large Language Models** [[Paper](https://arxiv.org/abs/2310.04560)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="Figures/Fatemi et al.jpg" /></p>
   <p align="center"><em>The designed prompts of Fatemi et al.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **GraphLLM: Boosting Graph Reasoning Ability of Large Language Model** [[Paper](https://arxiv.org/abs/2310.05845) | [Code](https://github.com/mistyreed63849/Graph-LLM)]
   <details close>
   <summary>GraphLLM</summary>
   <p align="center"><img width="75%" src="Figures/GraphLLM.jpg" /></p>
   <p align="center"><em>The framework of GraphLLM.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **Beyond Text: A Deep Dive into Large Language Model** [[Paper](https://arxiv.org/abs/2310.04944)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="Figures/Hu et al.jpg" /></p>
   <p align="center"><em>The designed prompts of Hu et al.</em></p>
   </details>
* (_2023.10_) [EMNLP' 2023] **MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter** [[Paper](https://arxiv.org/abs/2310.12798) | [Code](https://github.com/acharkq/MolCA)]
   <details close>
   <summary>MolCA</summary>
   <p align="center"><img width="75%" src="Figures/MolCA.jpg" /></p>
   <p align="center"><em>The framework of MolCA.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **GraphGPT: Graph Instruction Tuning for Large Language Models** [[Paper](https://arxiv.org/abs/2310.13023v1) | [Code](https://github.com/HKUDS/GraphGPT)]
   <details close>
   <summary>GraphGPT</summary>
   <p align="center"><img width="75%" src="Figures/GraphGPT.jpg" /></p>
   <p align="center"><em>The framework of GraphGPT.</em></p>
   </details>
* (_2023.10_) [EMNLP' 2023] **ReLM: Leveraging Language Models for Enhanced Chemical Reaction Prediction** [[Paper](https://arxiv.org/pdf/2310.13590.pdf) | [Code](https://github.com/syr-cn/ReLM)]
   <details close>
   <summary>ReLM</summary>
   <p align="center"><img width="75%" src="Figures/ReLM.jpg" /></p>
   <p align="center"><em>The framework of ReLM.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **LLM4DyG: Can Large Language Models Solve Problems on Dynamic Graphs?** [[Paper](https://arxiv.org/pdf/2310.17110.pdf)]
   <details close>
   <summary>LLM4DyG</summary>
   <p align="center"><img width="75%" src="Figures/LLM4DyG.jpg" /></p>
   <p align="center"><em>The framework of LLM4DyG.</em></p>
   </details>
* (_2023.10_) [Arxiv' 2023] **Disentangled Representation Learning with Large Language Models for Text-Attributed Graphs** [[Paper](https://arxiv.org/abs/2310.18152)]
   <details close>
   <summary>DGTL</summary>
   <p align="center"><img width="75%" src="Figures/DGTL.jpg" /></p>
   <p align="center"><em>The framework of DGTL.</em></p>
   </details>
* (_2023.11_) [Arxiv' 2023] **Which Modality should I use -- Text, Motif, or Image? : Understanding Graphs with Large Language Models** [[Paper](https://arxiv.org/abs/2311.09862)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="Figures/Das et al.jpg" /></p>
   <p align="center"><em>The framework of Das et al.</em></p>
   </details>
* (_2023.11_) [Arxiv' 2023] **InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery** [[Paper](https://arxiv.org/abs/2311.16208)]
   <details close>
   <summary>InstructMol</summary>
   <p align="center"><img width="75%" src="Figures/InstructMol.jpg" /></p>
   <p align="center"><em>The framework of InstructMol.</em></p>
   </details>
* (_2023.12_) [Arxiv' 2023] **When Graph Data Meets Multimodal: A New Paradigm for Graph Understanding and Reasoning** [[Paper](https://arxiv.org/pdf/2312.10372.pdf)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="Figures/Ai et al.jpg" /></p>
   <p align="center"><em>The framework of Ai et al.</em></p>
   </details>
* (_2024.02_) [Arxiv' 2024] **Let Your Graph Do the Talking: Encoding Structured Data for LLMs** [[Paper](https://arxiv.org/abs/2402.05862)]
   <details close>
   <summary>GraphToken</summary>
   <p align="center"><img width="75%" src="Figures/GraphToken.jpg" /></p>
   <p align="center"><em>The framework of GraphToken.</em></p>
   </details>
* (_2024.02_) [Arxiv' 2024] **Rendering Graphs for Graph Reasoning in Multimodal Large Language Models** [[Paper](https://arxiv.org/abs/2402.02130)]
   <details close>
   <summary>GITA</summary>
   <p align="center"><img width="75%" src="Figures/GITA.jpg" /></p>
   <p align="center"><em>The framework of GITA.</em></p>
   </details>
* (_2024.02_) [WWW' 2024] **GraphTranslator: Aligning Graph Model to Large Language Model for Open-ended Tasks** [[Paper](https://arxiv.org/abs/2402.07197) | [Code](https://github.com/alibaba/GraphTranslator)]
   <details close>
   <summary>GraphTranslator</summary>
   <p align="center"><img width="75%" src="Figures/GraphTranslator.jpg" /></p>
   <p align="center"><em>The framework of GraphTranslator.</em></p>
   </details>
* (_2024.02_) [Arxiv' 2024] **InstructGraph: Boosting Large Language Models via Graph-centric Instruction Tuning and Preference Alignment** [[Paper](https://arxiv.org/abs/2402.08785) | [Code](https://github.com/wjn1996/InstructGraph)]
   <details close>
   <summary>InstructGraph</summary>
   <p align="center"><img width="75%" src="Figures/InstructGraph.jpg" /></p>
   <p align="center"><em>The framework of InstructGraph.</em></p>
   </details>
* (_2024.02_) [Arxiv' 2024] **LLaGA: Large Language and Graph Assistant** [[Paper](https://arxiv.org/abs/2402.08170) | [Code](https://github.com/VITA-Group/LLaGA)]
   <details close>
   <summary>LLaGA</summary>
   <p align="center"><img width="75%" src="Figures/LLaGA.jpg" /></p>
   <p align="center"><em>The framework of LLaGA.</em></p>
   </details>
* (_2024.02_) [WWW' 2024] **Can GNN be Good Adapter for LLMs?** [[Paper](https://arxiv.org/abs/2402.12984)]
   <details close>
   <summary>GraphAdapter</summary>
   <p align="center"><img width="75%" src="Figures/graphadapter.jpg" /></p>
   <p align="center"><em>The framework of GraphAdapter.</em></p>
   </details>
* (_2024.02_) [Arxiv' 2024] **HiGPT: Heterogeneous Graph Language Model** [[Paper](https://arxiv.org/abs/2402.16024) | [Code](https://github.com/HKUDS/HiGPT)]
   <details close>
   <summary>HiGPT</summary>
   <p align="center"><img width="75%" src="Figures/HiGPT.jpg" /></p>
   <p align="center"><em>The framework of HiGPT.</em></p>
   </details>
* (_2024.02_) [Arxiv' 2024] **GraphWiz: An Instruction-Following Language Model for Graph Problems** [[Paper](https://arxiv.org/abs/2402.16029) | [Code](https://github.com/HKUDS/OpenGraph)]
   <details close>
   <summary>GraphWiz</summary>
   <p align="center"><img width="75%" src="Figures/GraphWiz.jpg" /></p>
   <p align="center"><em>The framework of GraphWiz.</em></p>
   </details>
* (_2024.03_) [Arxiv' 2024] **OpenGraph: Towards Open Graph Foundation Models** [[Paper](https://arxiv.org/abs/2403.01121) | [Code](https://github.com/nuochenpku/Graph-Reasoning-LLM)]
   <details close>
   <summary>OpenGraph</summary>
   <p align="center"><img width="75%" src="Figures/OpenGraph.jpg" /></p>
   <p align="center"><em>The framework of OpenGraph.</em></p>
   </details>

* (_2024.07_) [Arxiv' 2024] **GOFA: A Generative One-For-All Model for Joint Graph Language Modeling** [[Paper](https://arxiv.org/abs/2407.09709) | [Code](https://github.com/JiaruiFeng/GOFA)]
   <details close>
   <summary>GOFA</summary>
   <p align="center"><img width="75%" src="Figures/GOFA.jpg" /></p>
   <p align="center"><em>The framework of GOFA.</em></p>
   </details>
   
* (_2024.10_) [Arxiv' 2024] **Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?** [[Paper](https://arxiv.org/abs/2402.07140)]
   <details close>
   <summary>GraphDO</summary>
   <p align="center"><img width="75%" src="Figures/GraphDO.jpg" /></p>
   <p align="center"><em>The framework of GraphDO.</em></p>
   </details>
   
## GNN-LLM Alignment
* (_2020.08_) [Arxiv' 2020] **Graph-based Modeling of Online Communities for Fake News Detection** [[Paper](https://arxiv.org/abs/2008.06274) | [Code](https://github.com/shaanchandra/SAFER)]
   <details close>
   <summary>SAFER</summary>
   <p align="center"><img width="75%" src="Figures/SAFER.jpg" /></p>
   <p align="center"><em>The framework of SAFER.</em></p>
   </details>
* (_2021.05_) [NeurIPS' 2021] **GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph** [[Paper](https://arxiv.org/abs/2105.02605) | [Code](https://github.com/microsoft/GraphFormers)]
   <details close>
   <summary>GraphFormers</summary>
   <p align="center"><img width="75%" src="Figures/GraphFormers.jpg" /></p>
   <p align="center"><em>The framework of GraphFormers.</em></p>
   </details>
* (_2021.11_) [EMNLP' 2021] **Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries** 
  [[Paper](https://aclanthology.org/2021.emnlp-main.47/) | [Code](https://github.com/cnedwards/text2mol)]
   <details close>
   <summary>Text2Mol</summary>
   <p align="center"><img width="75%" src="Figures/Text2Mol.jpg" /></p>
   <p align="center"><em>The framework of Text2Mol.</em></p>
   </details>
* (_2022.07_) [ACL' 2023] **Hidden Schema Networks**
  [[Paper](https://arxiv.org/abs/2207.03777) | [Code](https://github.com/ramsesjsf/HiddenSchemaNetworks)]
   <details close>
   <summary>HSN</summary>
   <p align="center"><img width="75%" src="Figures/HSN.png" /></p>
   <p align="center"><em>The framework of HSN.</em></p>
   </details>
* (_2022.09_) [Arxiv' 2022] **A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language** 
  [[Paper](https://arxiv.org/abs/2209.05481) | [Code](https://github.com/BingSu12/MoMu)]
   <details close>
   <summary>MoMu</summary>
   <p align="center"><img width="75%" src="Figures/MoMu.jpg" /></p>
   <p align="center"><em>The framework of MoMu.</em></p>
   </details>
* (_2022.10_) [ICLR' 2023] **Learning on Large-scale Text-attributed Graphs via Variational Inference** 
  [[Paper](https://arxiv.org/abs/2210.14709) | [Code](https://github.com/AndyJZhao/GLEM)]
   <details close>
   <summary>GLEM</summary>
   <p align="center"><img width="75%" src="Figures/GLEM.jpg" /></p>
   <p align="center"><em>The framework of GLEM.</em></p>
   </details>
* (_2022.12_) [NMI' 2023] **Multi-modal Molecule Structure-text Model for Text-based Editing and Retrieval** 
  [[Paper](https://arxiv.org/abs/2212.10789) | [Code](https://github.com/chao1224/MoleculeSTM)]
   <details close>
   <summary>MoleculeSTM</summary>
   <p align="center"><img width="75%" src="Figures/MoleculeSTM.jpg" /></p>
   <p align="center"><em>The framework of MoleculeSTM.</em></p>
   </details>
* (_2023.04_) [Arxiv' 2023] **Train Your Own GNN Teacher: Graph-Aware Distillation on Textual Graphs** 
  [[Paper](https://arxiv.org/abs/2304.10668) | [Code](https://github.com/cmavro/GRAD)]
   <details close>
   <summary>GRAD</summary>
   <p align="center"><img width="75%" src="Figures/GRAD.jpg" /></p>
   <p align="center"><em>The framework of GRAD.</em></p>
   </details>
* (_2023.05_) [ACL' 2023] **PATTON : Language Model Pretraining on Text-Rich Networks** 
  [[Paper](https://arxiv.org/abs/2305.12268) | [Code](https://github.com/PeterGriffinJin/Patton)]
   <details close>
   <summary>Patton</summary>
   <p align="center"><img width="75%" src="Figures/Patton.jpg" /></p>
   <p align="center"><em>The framework of Patton.</em></p>
   </details>
* (_2023.05_) [Arxiv' 2023] **ConGraT: Self-Supervised Contrastive Pretraining for Joint Graph and Text Embeddings** 
  [[Paper](https://arxiv.org/abs/2305.14321) | [Code](https://github.com/wwbrannon/congrat)]
   <details close>
   <summary>ConGraT</summary>
   <p align="center"><img width="75%" src="Figures/ConGraT.jpg" /></p>
   <p align="center"><em>The framework of ConGraT.</em></p>
   </details>
* (_2023.07_) [Arxiv' 2023] **Prompt Tuning on Graph-augmented Low-resource Text Classification** 
  [[Paper](https://arxiv.org/abs/2307.10230) | [Code](https://github.com/WenZhihao666/G2P2-conditional)]
   <details close>
   <summary>G2P2</summary>
   <p align="center"><img width="75%" src="Figures/G2P2.jpg" /></p>
   <p align="center"><em>The framework of G2P2.</em></p>
   </details>
* (_2023.10_) [EMNLP' 2023] **GRENADE: Graph-Centric Language Model for Self-Supervised Representation Learning on Text-Attributed Graphs** 
  [[Paper](https://arxiv.org/abs/2310.15109) | [Code](https://github.com/bigheiniu/GRENADE)]
   <details close>
   <summary>GRENADE</summary>
   <p align="center"><img width="75%" src="Figures/GRENADE.jpg" /></p>
   <p align="center"><em>The framework of GRENADE.</em></p>
   </details>
* (_2023.10_) [WWW' 2024] **Representation Learning with Large Language Models for Recommendation** 
  [[Paper](https://arxiv.org/abs/2310.15950) | [Code](https://github.com/HKUDS/RLMRec)]
   <details close>
   <summary>RLMRec</summary>
   <p align="center"><img width="75%" src="Figures/RLMRec.jpg" /></p>
   <p align="center"><em>The framework of RLMRec.</em></p>
   </details>
* (_2023.10_) [EMNLP' 2023] **Pretraining Language Models with Text-Attributed Heterogeneous Graphs** 
  [[Paper](https://arxiv.org/abs/2310.12580) | [Code](https://github.com/Hope-Rita/THLM)]
   <details close>
   <summary>THLM</summary>
   <p align="center"><img width="75%" src="Figures/THLM.jpg" /></p>
   <p align="center"><em>The framework of THLM.</em></p>
   </details>

## Benchmarks

* (_2024.07_) [NeurIPS' 2024] **GLBench: A Comprehensive Benchmark for Graph with Large Language Models** [[Paper](https://arxiv.org/abs/2407.07457) | [Code](https://github.com/NineAbyss/GLBench)]
  
* (_2024.05_) [NeurIPS' 2024] **TEG-DB: A Comprehensive Dataset and Benchmark of Textual-Edge Graphs** [[Paper](https://arxiv.org/abs/2406.10310)][[Code](https://github.com/Zhuofeng-Li/TEG-Benchmark/tree/main)]
  

## Others

### LLM as Annotator

* (_2023.10_) [ICLR' 2024] **Label-free Node Classification on Graphs with Large Language Models (LLMs)** [[Paper](https://arxiv.org/abs/2310.18152) | [Code](https://github.com/CurryTang/LLMGNN)]
   <details close>
   <summary>LLM-GNN</summary>
   <p align="center"><img width="75%" src="Figures/LLM-GNN.png" /></p>
   <p align="center"><em>The framework of LLM-GNN.</em></p>
   </details>

* (_2024.09_) [NeurIPS' 2024] **Entity Alignment with Noisy Annotations from Large Language Models** [[Paper](https://arxiv.org/pdf/2405.16806) | [Code](https://github.com/chensyCN/llm4ea_official)]
   <details close>
   <summary>LLM4EA</summary>
   <p align="center"><img width="75%" src="Figures/LLM4EA.jpg" /></p>
   <p align="center"><em>The framework of LLM4EA.</em></p>
   </details>

### LLM as Controller

* (_2023.10_) [Arxiv' 2023] **Graph Neural Architecture Search with GPT-4** [[Paper](https://arxiv.org/abs/2310.01436)]
   <details close>
   <summary>GPT4GNAS</summary>
   <p align="center"><img width="75%" src="Figures/GPT4GNAS.jpg" /></p>
   <p align="center"><em>The framework of GPT4GNAS.</em></p>
   </details>

### LLM as Sample Generator

* (_2023.10_) [Arxiv' 2023] **Empower Text-Attributed Graphs Learning with Large Language Models (LLMs)** [[Paper](https://arxiv.org/abs/2310.09872)]
   <details close>
   <summary>ENG</summary>
   <p align="center"><img width="75%" src="Figures/ENG.jpg" /></p>
   <p align="center"><em>The framework of ENG.</em></p>
   </details>

### LLM as Similarity Analyzer

* (_2023.11_) [Arxiv' 2023] **Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs** [[Paper](https://arxiv.org/abs/2311.14324)]
   <details close>
   <summary>Framework</summary>
   <p align="center"><img width="75%" src="Figures/Sun et al.jpg" /></p>
   <p align="center"><em>The framework of Sun et al.</em></p>
   </details>

### LLM for Robustness 

- (_2024.05_) [Arxiv' 2024] **Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level** [[Paper](https://arxiv.org/abs/2405.16405)]
   <details close>
   <summary>Lei, et al.</summary>
   <p align="center"><img width="75%" src="Figures/Lei, et al.jpg" /></p>
   <p align="center"><em>The framework of Lei, et al..</em></p>
   </details>

- (_2024.08_) [Arxiv' 2024] **Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?** [[Paper](https://arxiv.org/abs/2408.08685)]
   <details close>
   <summary>LLM4RGNN</summary>
   <p align="center"><img width="75%" src="Figures/LLM4RGNN.jpg" /></p>
   <p align="center"><em>The framework of LLM4RGNN.</em></p>
   </details>

### LLM for Task Planning 

- (_2024.05_) [NeurIPS' 2024] **Can Graph Learning Improve Planning in LLM-based Agents?** [[Paper](https://arxiv.org/abs/2405.19119) | [Code](https://github.com/WxxShirley/GNN4TaskPlan)]
  <details close>
  <summary>GNN4TaskPlan</summary>
  <p align="center"><img width="75%" src="Figures/GNN4TaskPlan.jpg" /></p>
  <p align="center"><em>The definition of task planning and the proposed framework.</em></p>
  </details>


## Other Repos

We note that several repos also summarize papers on the integration of LLMs and graphs. However, we differentiate ourselves by organizing these papers leveraging a new and more granular taxonomy. We recommend researchers to explore some repositories for a comprehensive survey.

- [Awesome-Graph-LLM](https://github.com/XiaoxinHe/Awesome-Graph-LLM), created by [Xiaoxin He](https://xiaoxinhe.github.io/) from NUS.

- [Awesome-Large-Graph-Model](https://github.com/THUMNLab/awesome-large-graph-model), created by [Ziwei Zhang](https://zw-zhang.github.io/) from THU.

- [Awesome-Language-Model-on-Graphs](https://github.com/PeterGriffinJin/Awesome-Language-Model-on-Graphs), created by [Bowen Jin](https://peterjin.me/) from UIUC.

We highly recommend a repository that summarizes the work on **Graph Prompt**, which is very close to Graph-LLM.

- [Awesome-Graph-Prompt](https://github.com/WxxShirley/Awesome-Graph-Prompt), created by [Xixi Wu](https://wxxshirley.github.io/) from CUHK.


## Contributing

If you have come across relevant resources, feel free to open an issue or submit a pull request.

```
* (_time_) [conference] **paper_name** [[Paper](link) | [Code](link)]
   <details close>
   <summary>Model name</summary>
   <p align="center"><img width="75%" src="Figures/xxx.jpg" /></p>
   <p align="center"><em>The framework of model name.</em></p>
   </details>
```

## Cite Us

Feel free to cite this work if you find it useful to you!
```
@article{li2023survey,
  title={A Survey of Graph Meets Large Language Model: Progress and Future Directions},
  author={Li, Yuhan and Li, Zhixun and Wang, Peisong and Li, Jia and Sun, Xiangguo and Cheng, Hong and Yu, Jeffrey Xu},
  journal={arXiv preprint arXiv:2311.12399},
  year={2023}
}
```
===============================================================================================================================
