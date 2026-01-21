# CADET Framework Architecture

This document provides architectural diagrams and flow charts for the
CADET framework, illustrating the system design, data flows, and component interactions.

## 1. High-Level System Architecture

```mermaid
graph TB
subgraph Configuration
C1[Hydra Config]
C2[model config]
C3[data config]
C4[training config]
end

subgraph Pipeline Layer
P[Pipeline]
end

subgraph Data Layer
DL[DataLoader]
DS[HateSpeechDataset]
end

subgraph Model Layer
M[Model]
end

subgraph Training Layer
T[Trainer]
TO[Threshold Optimizer]
end

subgraph Evaluation Layer
E[Evaluator]
CM[Metrics]
VIZ[Visualization]
end

subgraph Output Layer
O1[checkpoints/]
O2[predictions/]
O3[metrics/]
O4[reports/]
end

C1 --> P
C2 --> M
C3 --> DL
C4 --> T

P --> DL
P --> M
P --> T
P --> E

DL --> DS
DL --> T
M --> T

T --> TO
T --> O1
T --> O2

E --> CM
E --> VIZ
E --> O3
E --> O4
```

## 2. Training Pipeline Flow

```mermaid
sequenceDiagram
participant U as User/Script
participant P as Pipeline
participant DL as DataLoader
participant T as Trainer
participant M as Model
participant E as Evaluator

U->>P: run_experiment()

P->>DL: load_data()
DL->>DL: Load HateSpeechDataset
DL->>DL: Tokenize data
DL-->>P: train, val, test datasets

P->>T: train()
T->>M: Load model
T->>T: Training loop
T->>T: Validate (optional)
T->>T: Save best checkpoint
T->>T: Optimize threshold (optional)
T-->>P: training_results

P->>T: inference()
T->>M: Load best checkpoint
T->>T: Run inference on test
T->>T: Save predictions
T-->>P: inference_results

P->>E: evaluate()
E->>E: Load predictions
E->>E: Compute metrics
E->>E: Generate visualizations
E->>E: Save reports
E-->>P: metrics

P-->>U: Complete results
```

## 3. Cross-Style Training Strategy

```mermaid
graph LR
subgraph Training Data
TS[Source Style<br/>explicit OR implicit]
TD[Train Split]
end

subgraph Validation Data
TT[Target Style<br/>opposite of source]
VD[Val Split]
end

subgraph Test Data
TT2[Target Style<br/>opposite of source]
TestD[Test Split]
end

subgraph Model
M[Model<br/>Training]
end

subgraph Optimization
THR[Threshold<br/>Optimization]
end

TS --> TD
TD --> M
TT --> VD
VD --> M
M --> THR
THR --> |optimal threshold| TestD
TT2 --> TestD
TestD --> |predictions| OUT[Results]
```

**Flow**:

1. Train on **source_style** (e.g., explicit hate speech)
2. Validate on **target_style** (e.g., implicit hate speech)
3. Optimize threshold on validation set
4. Test on **target_style** with fixed threshold

## 4. Data Flow Diagram

```mermaid
graph TD
subgraph Input
RAW[Raw Dataset<br/>Shuwan/cadet-datasets/]
end

subgraph DataLoader
LOAD[Load HateSpeechDataset]
SPLIT[Split by Style]
TOK[Tokenize]
end

subgraph Processing
TRAIN[Training Dataset]
VAL[Validation Dataset]
TEST[Test Dataset]
end

subgraph Format
FMT[Format for Model]
end

RAW --> LOAD
LOAD --> SPLIT

SPLIT --> |source-train| TRAIN
SPLIT --> |target-val| VAL
SPLIT --> |target-test| TEST

TRAIN --> TOK
VAL --> TOK
TEST --> TOK

TOK --> FMT
FMT --> OUT[Ready for Training]
```
