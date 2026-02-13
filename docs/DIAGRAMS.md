# Architecture Diagrams

## Diagram 1: System Context

```mermaid
graph TB
    subgraph External
        A[User/Tester]
        B[Image Files]
        C[Config Files]
    end
    
    subgraph "Vision Router Core"
        D[main.py CLI]
        E[Pipeline Orchestrator]
    end
    
    subgraph Output
        F[JSON Results]
        G[Console Output]
    end
    
    A -->|Commands| D
    B -->|Images| D
    C -->|Thermal Config| D
    D --> E
    E --> F
    E --> G


    graph LR
    subgraph Input
        A[Image File]
    end
    
    subgraph "Pipeline"
        B[Resolution Scaler]
        C[Security Filter]
        D[Vision Encoder]
        E[Intent Router]
    end
    
    subgraph Output
        F[JSON Result]
    end
    
    A --> B
    B -->|Resized Image| C
    C -->|Sanitized Image| D
    D -->|Class Probabilities| E
    E -->|Intent| F
    
    T[Thermal State] -.->|Config| B


    sequenceDiagram
    participant User
    participant CLI as main.py
    participant RS as ResolutionScaler
    participant SF as SecurityFilter
    participant VE as VisionEncoder
    participant IR as IntentRouter
    
    User->>CLI: --image doc.jpg --thermal serious
    CLI->>RS: get_resolution(SERIOUS)
    RS-->>CLI: (256, 256)
    CLI->>RS: resize_image(image, SERIOUS)
    RS-->>CLI: resized_image
    CLI->>SF: scan(resized_image, "doc.jpg")
    SF-->>CLI: SecurityResult(is_safe=True)
    CLI->>VE: encode(image)
    VE-->>CLI: EncodingResult(class="notebook")
    CLI->>IR: route(695, "notebook")
    IR-->>CLI: INTENT_A_PRACTICAL_GUIDANCE
    CLI->>User: JSON Output



stateDiagram-v2
    [*] --> NOMINAL: Default
    
    NOMINAL --> FAIR: Temperature rising
    FAIR --> SERIOUS: Temperature high
    SERIOUS --> CRITICAL: Overheating
    
    CRITICAL --> SERIOUS: Cooling
    SERIOUS --> FAIR: Cooling
    FAIR --> NOMINAL: Normal temp
    
    NOMINAL: Resolution 768x768
    FAIR: Resolution 512x512
    SERIOUS: Resolution 256x256
    CRITICAL: Resolution 256x256




flowchart TD
    A[Input Image] --> B{Sensitive Filename?}
    B -->|Yes| C[Apply Gaussian Blur]
    B -->|No| D{OCR Enabled?}
    C --> D
    D -->|Yes| E[Scan for Text]
    D -->|No| G[Output Image]
    E --> F{Adversarial Keywords?}
    F -->|Yes| H[Mask Text Regions]
    F -->|No| G
    H --> G
    G --> I[SecurityResult]


    flowchart TD
    A[Class ID + Name] --> B{ID in Mapping?}
    B -->|Yes| C[Return Mapped Intent]
    B -->|No| D{Name has A keywords?}
    D -->|Yes| E[INTENT_A: Practical]
    D -->|No| F{Name has C keywords?}
    F -->|Yes| G[INTENT_C: Creative]
    F -->|No| H{Name has B keywords?}
    H -->|Yes| I[INTENT_B: Discovery]
    H -->|No| J[Default: INTENT_B]


    