# Product Requirements Document
## AarohAI — AI-Powered Criminal Network Analysis System
**SIH 2026 | PS ID: SIH26189 | Ministry of Home Affairs (MHA)**

Version 1.0 · Prepared for Smart India Hackathon 2026

---

## 1. Problem Understanding

Law enforcement agencies today sit on enormous, disconnected volumes of data — FIRs, chargesheets, Call Detail Records (CDR), Internet Protocol Detail Records (IPDR), financial transactions, prison visitor logs, vehicle registrations, and open-source/social media intelligence. The people behind organised crime (drug syndicates, cybercrime rings, human trafficking networks, gang federations) deliberately obscure their connections across these silos. Manual link analysis by an investigator is slow, inconsistent, and misses non-obvious relationships (a shared financier, a common lawyer, a pattern of co-travel, a burner phone reused across cases).

**Core ask:** a system that ingests heterogeneous crime data, constructs a relationship graph of people/places/assets/events, surfaces hidden criminal networks and their key players, predicts likely undiscovered links, and presents all of this to an investigator in a way that is explainable, auditable, and usable in court.

**What actually makes this hard (and where most student teams lose marks):**
- Data is multilingual (Hindi/regional language FIRs), unstructured, and inconsistently formatted across states.
- Names and identities are noisy — aliases, transliteration variants, partial addresses.
- A single LLM "chatbot over crime data" demo is not defensible for MHA-grade use: no explainability, no audit trail, no offline/on-prem story, no legal admissibility.
- Judges have seen "graph + chatbot" demos before (Neo4j + GPT is a known pattern in this space). Winning requires provable investigative value, not just a pretty graph.

This PRD is written to produce a solution that is technically demo-able in 36–45 hours **and** credible as a production MHA system — which is what separates a 6/10 SIH submission from a 10/10 one.

---

## 2. Vision Statement

> Give every investigating officer in India the analytical capability of an entire intelligence cell — turning weeks of manual link-analysis into minutes, with every AI conclusion traceable back to the evidence that produced it.

---

## 3. Goals & Success Metrics

| Goal | Metric |
|---|---|
| Reduce time to identify a criminal network from raw case data | From days (manual) to minutes (demo: <5 min for a 500-node synthetic dataset) |
| Surface hidden/non-obvious links | Recall of "planted" hidden relationships in synthetic test data ≥ 80% |
| Identify key players / kingpins correctly | Precision@5 on centrality-ranked suspects vs. ground truth ≥ 85% |
| Explainability | 100% of AI-surfaced links show a provenance trail (source document + confidence score) |
| Usability for non-technical officers | Task completion (find a network, export a report) without training, in a usability test |
| Legal/compliance readiness | Full audit log of every query, every access, every AI inference |

---

## 4. Users & Personas

1. **Investigating Officer (IO)** — needs to quickly see who is connected to a suspect, within N hops, with evidence.
2. **SP / Crime Branch Analyst** — needs district/state-level pattern views, hotspot and gang-mapping.
3. **Cyber Cell Analyst** — needs digital footprint correlation (IPDR, financial, social media).
4. **Prosecutor** — needs a clean, court-presentable network diagram + evidence chain for chargesheets.
5. **NCRB / State CID Admin** — needs system administration, data source management, cross-state network correlation.
6. **Field Officer (mobile)** — needs offline/low-connectivity lookup of a person's known associates.

---

## 5. Functional Requirements

### 5.1 Data Ingestion & Preparation
- FR1: Ingest structured data (CDR, IPDR, financial transaction CSVs, vehicle/property records) via templated connectors.
- FR2: Ingest unstructured data (FIR text, chargesheets, witness statements) in English + Hindi + regional languages.
- FR3: OCR pipeline for scanned FIRs/handwritten records.
- FR4: Deduplication and entity resolution across sources (same person, different spelling/alias).

### 5.2 Entity & Relationship Extraction
- FR5: Named Entity Recognition for persons, organisations, locations, vehicles, phone numbers, financial accounts, weapons.
- FR6: Relationship extraction (co-accused, family, financial transaction, communication, co-location/co-travel, prior association).
- FR7: Alias resolution using phonetic + embedding similarity (e.g., "Md. Rafique" vs "Rafiq bhai").

### 5.3 Graph Construction & Analysis
- FR8: Build a property graph (Neo4j) of entities and typed relationships with timestamps and source provenance on every edge.
- FR9: Run network analysis: degree/betweenness/eigenvector centrality (kingpin detection), Louvain/Leiden community detection (gang/cell identification), shortest-path and k-hop queries.
- FR10: Hidden-link prediction using a Graph Neural Network (GraphSAGE/link prediction) to suggest **probable but undocumented** connections, always flagged as "predicted — not confirmed" with a confidence score.
- FR11: Temporal graph view — how a network evolved over time (new members, broken links, reformed cells after arrests).

### 5.4 Investigation & Query Interface
- FR12: Natural-language query ("show everyone within 2 hops of X who has a financial link") translated to Cypher, with generated answer *grounded* in retrieved subgraph — never a free-form hallucinated answer.
- FR13: Interactive graph visualization with filter/expand/collapse, geo-map overlay of events, timeline scrubber.
- FR14: "Explain this link" — click any edge/inference to see the exact source document, sentence, and extraction confidence.

### 5.5 Reporting & Output
- FR15: One-click generation of a court-ready network diagram + annexure document (chargesheet-ready PDF).
- FR16: Case file export with full evidence chain and audit trail.

### 5.6 Alerting & Monitoring
- FR17: Real-time alert when new incoming data (e.g. a new FIR, a new CDR dump) matches or extends an existing monitored network.
- FR18: Risk/recidivism scoring for known offenders based on network position and history (advisory only, never automated decision-making).

### 5.7 Security, Access & Compliance
- FR19: Role-based access control (RBAC) — an officer only sees cases/jurisdictions they're authorised for.
- FR20: Full audit trail — every view, query, export logged immutably (append-only log / hash-chained).
- FR21: PII redaction controls for any output that leaves the secure perimeter.
- FR22: Human-in-the-loop confirmation required before any AI-suggested link is treated as "confirmed" in a case file.

---

## 6. Non-Functional Requirements

- **Security-first, on-prem capable:** must be deployable within NIC/MeghRaj government data centers, air-gapped if required. No mandatory external API calls for core inference.
- **Data sovereignty:** all case data stays within India-hosted infrastructure; no case-identifiable data sent to third-party cloud LLMs unless explicitly permitted and redacted.
- **Scalability:** graph of 10M+ nodes / 50M+ edges, sub-2-second query response for 3-hop traversals.
- **Multilingual:** Hindi + at least 3 regional languages (configurable) for NER/extraction.
- **Explainability:** every AI output must be traceable to source evidence (non-negotiable for legal admissibility).
- **Auditability:** tamper-evident logs (hash chaining or WORM storage).
- **Availability:** 99.5% uptime target for production; graceful degradation to read-only graph view if AI services are down.
- **Interoperability:** exposes APIs compatible with CCTNS (Crime and Criminal Tracking Network & Systems) and ICJS (Inter-operable Criminal Justice System) data formats.

---

## 7. Multi-Agent Architecture (the core technical differentiator)

Why multi-agent instead of one big model: each investigative task (extraction, graph reasoning, prediction, report writing, compliance checking) needs different tools, different data access levels, and different failure modes. A single monolithic LLM pipeline can't be audited step-by-step and can't be selectively run on-prem vs. cloud. A multi-agent design lets you swap, audit, and scale each capability independently — which is also exactly how you'd defend it to judges as "production thinking," not just a hackathon trick.

### 7.1 Agent Roster

| Agent | Responsibility | Key tech |
|---|---|---|
| **Orchestrator Agent** | Receives a task (new data batch, or an investigator's question), plans which agents to invoke and in what order, maintains shared state, and never lets raw output reach the user without passing through the Compliance Agent | LangGraph (stateful agent graph), a routing LLM |
| **Ingestion Agent** | Parses incoming files (PDF/scan/CSV/text), normalises formats, hands off to extraction | Python, Apache Tika, OCR (Tesseract/PaddleOCR) |
| **Entity Resolution Agent** | NER across languages, alias/name matching, dedup, assigns stable entity IDs | IndicBERT/IndicNER, sentence-embedding similarity (multilingual-e5), fuzzy/phonetic matching (Soundex/Metaphone adapted for Indian names) |
| **Graph Builder Agent** | Converts resolved entities + relationships into graph writes, manages versioning of the graph (who added what, when) | Neo4j + Graph Data Science library, Cypher |
| **Network Analysis Agent** | Runs centrality, community detection, path-finding; ranks likely kingpins and sub-groups | Neo4j GDS (Louvain/Leiden, PageRank, betweenness) |
| **Predictive/Link-Prediction Agent** | GNN-based hidden-link prediction; recidivism/risk scoring; always outputs a confidence score and marks results as "predicted" | PyTorch Geometric, GraphSAGE |
| **NL Query / RAG Agent** | Converts investigator's natural-language question into a Cypher query over the *current, scoped* subgraph, retrieves subgraph facts, and generates a grounded natural-language answer with citations to source documents | LLM (on-prem Llama/Mistral fine-tune for sensitive queries, or Claude via API for non-sensitive/dev environments) + text-to-Cypher + RAG over a vector store of source docs |
| **Explainability Agent** | Attaches a provenance trail (source doc, sentence span, extraction confidence, model version) to every entity/edge/prediction shown to a user | Metadata store, lineage tracking |
| **Compliance & Guardrail Agent** | Checks RBAC before returning any result, redacts PII where required, enforces "predicted vs confirmed" labelling, blocks any hallucinated/unsupported claim from reaching the UI, writes the audit log entry | Policy engine (OPA — Open Policy Agent), custom validators |
| **Report Generation Agent** | Produces the court-ready PDF/annexure: network diagram + evidence table + case summary | Templated generation (Python + a document-generation library), grounded strictly in graph + explainability data |
| **Alert/Monitoring Agent** | Watches the ingestion stream for new data that touches a monitored network or matches a saved pattern; triggers notifications | Event stream (Kafka/Redis Streams), rule + anomaly triggers |

### 7.2 How they collaborate

1. New data arrives → **Orchestrator** hands it to **Ingestion Agent** → **Entity Resolution Agent** → **Graph Builder Agent** updates Neo4j.
2. **Network Analysis Agent** and **Predictive Agent** run (batch, scheduled or triggered) to refresh centrality scores and hidden-link suggestions.
3. Investigator asks a question in the UI → **Orchestrator** routes to **NL Query/RAG Agent**, which queries the graph + vector store, drafts an answer.
4. Every answer, before display, passes through **Explainability Agent** (attach provenance) and **Compliance Agent** (RBAC + redaction + audit log write). Only then does the Orchestrator return it to the UI.
5. If the investigator requests a report, **Report Generation Agent** assembles it from the same provenance-backed graph data — never from free LLM generation alone.
6. **Alert Agent** runs continuously in the background and can independently trigger the Orchestrator when new data matches a watch pattern.

This "nothing reaches the user unchecked" design is what makes the system MHA-defensible, not just technically impressive — and it's an easy thing to say clearly in your judging round when asked "how do you prevent the AI from hallucinating a false accusation."

### 7.3 Why this beats a single-LLM demo
- **Auditable:** every agent's input/output is logged separately — you can show a judge exactly which agent produced a given claim.
- **Swappable for compliance:** the NL Query Agent's LLM can be swapped from a cloud API (fast to build for the hackathon) to a fully on-prem model (production requirement) without touching the rest of the system.
- **Fails safe:** if the Predictive Agent is wrong or unavailable, the rest of the system (confirmed graph + analysis) still works — no single point of failure in the "intelligence" layer.
- **Matches how MHA actually procures systems:** modular, independently upgradable components map cleanly onto a phased govt rollout instead of a black box.

---

## 8. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | Next.js (React), Tailwind CSS | fast to build, good for dashboards |
| Graph visualization | Cytoscape.js / Sigma.js | handles large graphs, good interactivity |
| Geospatial | Mapbox / deck.gl | event/location overlays |
| Backend API | FastAPI (Python) | async, fast, great for ML integration |
| Agent orchestration | LangGraph (or CrewAI as alternative) | stateful, auditable multi-agent workflows |
| LLM | Hybrid: on-prem fine-tuned Llama/Mistral for sensitive case data; cloud LLM (e.g. Claude API) for dev/demo and non-sensitive augmentation | data sovereignty + fast hackathon build |
| Graph database | Neo4j + Graph Data Science library (or JanusGraph for extreme scale) | native graph algorithms, Cypher query language |
| Vector store | Qdrant or pgvector | RAG over source documents |
| GNN | PyTorch Geometric (GraphSAGE) | link prediction |
| NLP / NER | spaCy, IndicNER, IndicBERT, multilingual-e5 embeddings | Indian-language support |
| Messaging / streaming | Kafka or Redis Streams | ingestion pipeline + alerting |
| Structured storage | PostgreSQL | case metadata, users, audit logs |
| Document/object storage | MinIO (S3-compatible) | source documents, scanned FIRs |
| Full-text search | Elasticsearch / OpenSearch | document search |
| Auth | Keycloak | RBAC, SSO-ready for govt identity systems |
| Secrets | HashiCorp Vault | key management |
| Policy/guardrails | Open Policy Agent (OPA) | RBAC + redaction enforcement |
| Observability | Prometheus + Grafana, ELK stack | monitoring + audit log analytics |
| Deployment | Docker + Kubernetes | on-prem/govt-cloud (MeghRaj) deployable |
| Mobile (field officer) | React Native / Flutter with offline-first sync | low-connectivity field use |

---

## 9. Data Flow (high level)

1. **Ingest** → raw FIRs/CDRs/financial records/social data land in object storage, queued via Kafka.
2. **Extract** → Ingestion + Entity Resolution Agents parse and normalise, write resolved entities to a staging table.
3. **Graph** → Graph Builder Agent writes nodes/edges to Neo4j with provenance metadata on every edge.
4. **Analyze** → Network Analysis + Predictive Agents run scheduled/triggered jobs, write scores back as node/edge properties.
5. **Serve** → FastAPI backend exposes REST/GraphQL over the graph + analysis results to the Next.js frontend.
6. **Query** → Investigator's NL question flows through Orchestrator → RAG/Query Agent → Explainability Agent → Compliance Agent → UI.
7. **Report/Alert** → Report Agent and Alert Agent operate off the same provenance-backed data.

---

## 10. Demo Scenario (what you actually show judges)

Use a **synthetic but realistic** dataset (never real case data) — e.g. 300–500 entities, 5–6 planted "hidden" networks, some deliberately obscured links (shared financier, common phone reused under different names, co-travel patterns).

**Live demo flow:**
1. Show raw messy input (a scanned FIR excerpt, a CDR CSV, a financial transaction list) — emphasize real-world messiness.
2. Trigger ingestion → within seconds, show the graph populating live.
3. Ask in natural language: *"Who are the top 5 most influential people connected to [suspect] within 3 hops, and why?"* — system answers with a ranked list, graph highlight, and citation to source documents for each claim.
4. Click "Explain this link" on a surprising edge → show the exact source sentence and confidence score.
5. Trigger the Predictive Agent: *"Show me likely undiscovered connections"* → system shows 2–3 GNN-predicted links, clearly labelled **"predicted, not confirmed."**
6. Generate a one-click court-ready PDF report.
7. Show the audit log — every action just performed is logged immutably.
8. (Bonus, if time permits) Show the mobile app doing an offline lookup, then syncing.

This sequence hits every judging criterion: technical depth, real-world feasibility, explainability, and social/legal impact — in under 5 minutes.

---

## 11. Innovation / USP (what should differentiate you from other teams doing the same PS)

1. **Provenance-first design** — every single claim the AI makes is traceable to a source document and a confidence score. Most competing solutions in this space stop at "we built a graph + chatbot."
2. **"Predicted vs Confirmed" distinction enforced structurally**, not just in UI copy — the Compliance Agent will not let an unconfirmed GNN prediction be labelled or exported as fact. This directly addresses the single biggest real-world objection to AI in policing (false accusation risk) and is a strong point to make explicitly to judges.
3. **Hybrid on-prem/cloud LLM architecture** — addresses MHA's actual, real deployment constraint (data can't leave government infrastructure) instead of ignoring it like most hackathon prototypes.
4. **Multilingual-first NER**, because FIRs in India are frequently filed in the regional language, not English — most demo systems silently assume English-only input and this is an easy point to lose if a judge asks about it.
5. **Legal admissibility features** — hash-chained audit logs and evidence-chain export, aimed at chargesheet use, not just an internal dashboard.
6. **Cross-state federated correlation (roadmap item)** — a privacy-preserving way for two states to discover that their separately investigated suspects are connected, without sharing raw case data (federated learning / secure multi-party computation on graph summaries). Mention this as a differentiator even if you don't build it — it shows you understand the real inter-state intelligence-sharing problem NCRB struggles with today.
7. **Mobile offline-first field app** for officers without reliable connectivity — grounded in a very real operational constraint in rural/border policing.

---

## 12. What to Actually Build in 36–45 Hackathon Hours (scope discipline)

Being honest about hackathon time is itself a differentiator — judges can tell when a team overclaims. Build this in order; **stop and demo well** rather than half-build everything:

**Must build (core, ~60% of time):**
- Synthetic dataset generator (this alone saves you from a legal/ethics objection about using real data)
- Ingestion → NER → Entity Resolution → Neo4j graph pipeline for at least 2 data types (FIR text + CDR)
- Graph visualization UI with centrality-based highlighting and community detection
- NL Query Agent with grounded, cited answers (this is your single most impressive demo moment — prioritize it)
- Explainability click-through (source doc + confidence)
- Basic RBAC login

**Should build (~25% of time):**
- GNN-based hidden link prediction (even a simple GraphSAGE trained on the synthetic graph is enough to demo "predicted, not confirmed" links)
- One-click PDF report generation
- Audit log viewer

**Nice to have if time remains (~15%):**
- Alert agent with a live-triggered demo ("new FIR just came in, watch the network update")
- Mobile offline demo (can be a lightweight React Native screen, doesn't need full offline sync working)
- Multilingual NER on a second language sample, even just Hindi

**Explicitly describe as roadmap, don't try to build:** federated cross-state learning, full CCTNS/ICJS integration, production-grade on-prem LLM fine-tuning. Say this confidently in the PPT — "phase 2/3" — rather than pretending it's built.

---

## 13. Team Roles (suggested, for a 6-member SIH team)

| Role | Focus |
|---|---|
| Team Lead / Backend | FastAPI, orchestrator, agent wiring |
| Graph Engineer | Neo4j schema, Cypher, GDS algorithms |
| ML/NLP Engineer | NER, entity resolution, GNN link prediction |
| LLM/Agent Engineer | LangGraph agents, RAG, prompt design, guardrails |
| Frontend Engineer | Next.js dashboard, graph visualization |
| Presenter / PM / Domain research | Problem framing, synthetic dataset design, PPT, demo script, MHA policy context (CCTNS/ICJS/DPDP Act) |

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM hallucinates a false link/accusation | Compliance Agent blocks any unsupported claim; grounded RAG only; "predicted vs confirmed" enforced structurally |
| Real crime data privacy/ethics concerns | Use only synthetic data for the hackathon; document a clear data governance plan for production |
| Judges question feasibility of on-prem LLM | Show the architecture is LLM-agnostic — same agent graph, swap the model |
| Graph performance at scale | Use Neo4j GDS's native algorithms (not naive Python graph traversal); mention JanusGraph/sharding as the scale-out path |
| Multilingual NER accuracy | Use pretrained IndicNER/IndicBERT rather than building from scratch; be honest about current accuracy in the demo |
| Time runs out before full build | Follow the scoped priority list in Section 12; a polished narrow demo beats a broken broad one |

---

## 15. Compliance & Legal Considerations (mention these explicitly to judges — it signals maturity)

- **DPDP Act 2023** — data minimisation, purpose limitation, and access logging for any personal data processed.
- **IT Act 2000** — evidentiary standards for electronic records (Section 65B relevance for court-admissible exports).
- **Chain of custody** — every piece of evidence shown in a report must be traceable to its original source and unaltered.
- **Human-in-the-loop mandate** — the system never makes an arrest/charge recommendation; it surfaces leads for a human investigator to verify.

---

## 16. Roadmap Beyond the Hackathon

- **Phase 1 (0–3 months):** Pilot with one state police cyber cell on synthetic + sanitised historical data; refine NER for 2–3 regional languages.
- **Phase 2 (3–9 months):** On-prem LLM fine-tuning, CCTNS/ICJS API integration, full RBAC + audit hardening, mobile field app.
- **Phase 3 (9–18 months):** Cross-state federated correlation, recidivism risk scoring validated against outcomes, national rollout via NCRB.

---

## 17. One-line Pitch (for your PPT title slide)

> "AarohAI turns scattered FIRs, call records, and financial data into a living, explainable map of criminal networks — so investigators find the kingpin in minutes, not months, with every conclusion backed by evidence a court can trust."

---

*Prepared as a working PRD for SIH 2026, PS ID SIH26189. Adjust the team/tech names and demo dataset before submission.*
