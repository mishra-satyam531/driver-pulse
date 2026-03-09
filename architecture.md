# System Architecture: Connecting Chaos to Clarity

> *Driving generates gigabytes of noisy entropy daily. Our architecture acts as the silencer, reducing digital noise into sparse, human-readable insights.*

---

## 1. Architecture Diagram (Data Flow & Node Processing)

```mermaid
flowchart TD
    %% Mobile / Client Processing
    subgraph Edge [The Driver's Device (Edge Node)]
        direction TB
        S1[IMU Sensor Array\n100Hz Accelerometer]
        S2[Microphone Array\n32kHz Audio]
        Filter[DSP Filter & Envelope Extractor]
        Buffer[(SQLite Offline Queue)]
        UI[Streamlit Dashboard\nRendered React View]
        TTS[On-Device TTS Engine]
    end

    %% Cloud / Backend Engine
    subgraph Cloud [Cloud Intelligence Layer]
        direction TB
        Gateway[API Gateway\nRate Limiter]
        Lake[(Time-Series Data Lake)]
        StressCore{Physics Logic Engine\nWindowing & Joining}
        VeloCore{Earnings Velocity\nRandom Forest Predictor}
        LLM[GenAI Insight LLM\nGroq Llama 3]
        DB[(Consolidated Insights DB)]
    end

    %% Communication Flow
    S1 & S2 --> Filter
    Filter -->|Decoupled Batches (Resampled)| Buffer
    Buffer -->|RESTful Sync (Connectivity-Aware)| Gateway
    Gateway --> Lake
    
    Lake --> StressCore
    Lake --> VeloCore
    StressCore & VeloCore --> DB
    
    DB -->|Medium/High Severity Events| LLM
    LLM -->|Empathetic Text Scripts| DB

    %% Read Operations
    DB -->|Fetch JSON Array| UI
    UI -->|Localization Request| TTS
```

---

## 2. The Architectural "Why": Extreme Constraints Driving the Design

Building software for moving vehicles introduces constraints that typical web architecture ignores. Our backend design prioritizes **Driver Safety, Deep Privacy, and Battery Survival**.

### 2.1 The Connectivity Problem (Network Resilience)
**The Constraint:** Cars routinely drop down to 2G or entirely lose cellular connectivity in parking garages, tunnels, and rural borders.
**The Architecture Solution:** **Decoupled Offline Queues.**
We designed the client to cache and batch resampled telematics locally. If the API is unreachable, arrays are held in memory. When reconnections happen, the `Time-Series Data Lake` successfully ingests them. Crucially, the Backend `Physics Logic Engine` uses absolute Unix timestamps (`pd.merge_asof` joins) so that inherently out-of-order, delayed events still successfully generate contextually accurate stress insights.

### 2.2 The Privacy Problem (Ethical Audio Surveillance)
**The Constraint:** You cannot record and upload a driver's live conversation or a passenger's argument. Period.
**The Architecture Solution:** **Physical Minimization Before Cloud.**
Our Edge Node architecture forces the audio path through a hard DSP Filter instantly upon capture. **The cloud never touches raw audio**. The `DSP Filter & Envelope Extractor` locally calculates a rolling mean in decibels (`dB`), creating a one-dimensional, anonymous numerical array. What the Cloud receives is `"85.2 dB"` not `"The passenger yelled at me."` The system mathematically cannot spy.

### 2.3 The Battery Problem (Processing Offloading)
**The Constraint:** Drivers need their phone screen constantly on for Uber apps and GPS. High-frequency 100Hz continuous polling instantly destroys battery life.
**The Architecture Solution:** **Aggressive Batching & Edge/Cloud Division.**
Instead of a live, streaming WebSocket connection calculating rolling standard deviations on an iPhone, the device sends small batched chunks out every 30 seconds. The **Random Forest Predictor** (Earnings), **GenAI LLM** (Insights), and **Physics Logic Engine** (Stress) are all computed on high-performance Cloud hardware. The device simply renders the final JSON snapshot via the Streamlit UI.

### 2.4 The Dashboard Layer (Glanceability)
**The Constraint:** A driver who is actively driving should not be reading complex graphs.
**The Architecture Solution:** **Asynchronous Rendering & TTS.**
The dashboard reads precomputed JSON objects from the Insights DB instantly. Every single flag comes with a simple `🔊` icon linked natively to the On-Device TTS Engine. This makes the UI not just readable, but audibly accessible so a driver can listen to their progress safely.
