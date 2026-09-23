# TRINETRA-X
## The Unified Investigation Intelligence Grid
**Smart India Hackathon 2026 | Problem Statement ID: SIH26189 | Ministry of Home Affairs**

*A merged solution combining TRINETRA's field-ready focus with AarohAI's production-grade depth*

---

## 1. Why This Version Exists

TRINETRA and AarohAI attack the same problem — fragmented investigative data (FIRs, CDRs, IPDRs, financial records, vehicle/location logs) that investigators today cross-reference by hand — and land on the same core idea: build a live knowledge graph, add GNN-based hidden-link prediction, and wrap every AI finding in provenance and confidence scores so it survives courtroom and judge scrutiny.

Each had a real gap the other one closed:

| Gap in TRINETRA | Filled by AarohAI |
|---|---|
| No multilingual handling — Indian FIRs are frequently filed in Hindi/regional languages | Multilingual NER (IndicBERT/IndicNER) built in from the start |
| Single AI pass with no structural guardrail against hallucinated claims reaching the user | Compliance/Guardrail Agent that blocks unsupported claims before display |
| No measurable success criteria | Concrete metrics: recall, precision@5, query latency, explainability coverage |
| No grounding in specific Indian legal frameworks | Explicit DPDP Act 2023, IT Act §65B, CCTNS/ICJS interoperability |

| Gap in AarohAI | Filled by TRINETRA |
|---|---|
| 11-agent architecture not buildable by 6 people in 36–45 hours — high execution risk | Simpler, demo-tight architecture that's realistically finishable |
| No cross-agency/cross-border story despite mentioning transnational crime | Federated query layer exchanging only hashed matches, never raw PII |
| No field/offline story for rural or border deployment | Offline-first mobile companion caching the active case subgraph |
| Roadmap items stated but not honestly triaged against hackathon time | Explicit "build vs. roadmap" split with a heuristic stand-in for the GNN |

TRINETRA-X keeps TRINETRA's disciplined, demo-able core and grafts on AarohAI's production-grade safeguards and Indian-specific realism — without inheriting AarohAI's overbuild risk.

---

## 2. Problem Statement (SIH26189)

Investigative data — case records, communications, financial transactions, vehicle/location logs, and evidence files — is scattered across disconnected authorized sources, often multilingual and inconsistently formatted. Investigators manually cross-reference these to trace how a suspect, number, account, or location connects to a case — a process that doesn't scale past 2–3 degrees of connection and misses non-obvious links (a shared financier, a reused burner phone, a common lawyer, a pattern of co-travel).

The system must surface investigative leads — not determine legal guilt — and every AI-generated finding must be explainable, auditable, and admissible.

---

## 3. Solution Overview

TRINETRA-X is a case-based investigation intelligence platform. An investigator opens a case, asks a natural-language question ("How is Rahul Sharma connected to Operation Blackout?"), and the system resolves entities, traverses a live knowledge graph across authorized sources, runs GNN-based hidden-link prediction, and returns confidence-scored findings — each paired with mandatory counter-evidence and a full provenance trail back to source records — before anything reaches the investigator's screen.

### 3.1 What Makes It Different

| Capability | Standard Approach | TRINETRA-X |
|---|---|---|
| Relationship discovery | Traverses only known/recorded links | GNN link-prediction surfaces probable undocumented connections, always labelled "predicted — not confirmed" |
| Evidence trust | AI summary with no audit trail | Every graph edge carries a hash-anchored provenance record (source ID + timestamp + retrieval hash) |
| AI reliability | One-sided AI "finding" | Compliance Guardrail structurally blocks any unsupported claim; mandatory counter-evidence on every finding |
| Language coverage | English-only extraction (a common silent failure point for Indian systems) | Multilingual NER/NLP (Hindi + regional languages) for FIRs and witness statements |
| Deployment scope | Single-agency, single-database tool | Federated query layer exchanging only hashed matches — never raw PII — across agencies or countries |
| Field usability | Control-room only, always-online | Offline-first mobile companion caching the active case subgraph |
| Legal grounding | Generic "compliance" claims | Explicit DPDP Act 2023, IT Act §65B admissibility, CCTNS/ICJS-compatible APIs |
| Success criteria | Impact bullets, no numbers | Measurable targets (below) that judges can actually score against |

---

## 4. End-to-End Workflow

1. Investigator opens/creates a case and submits a natural-language query.
2. **Entity Resolution** normalizes the query against the authorized data store — including alias/transliteration matching (phonetic + embedding similarity) so "Md. Rafique" and "Rafiq bhai" resolve to the same node — and asks for clarification if the query is ambiguous.
3. **Traversal Service** runs a depth-limited, cycle-protected exploration of authorized relationships, streaming nodes and edges live to the investigator's screen as the graph builds.
4. **Link-Prediction Service** (GNN) proposes probable undocumented connections — shared intermediaries, overlapping geofences, indirect financial ties — always tagged "predicted, not confirmed."
5. **Pattern Analysis** runs one final AI pass over the completed graph, producing structured findings: title, explanation, confidence score, and mandatory counter-evidence.
6. **Explainability layer** attaches a provenance trail to every claim — source document, sentence span, extraction confidence.
7. **Compliance Guardrail** checks RBAC, enforces "predicted vs. confirmed" labelling, redacts PII where required, and writes an immutable audit-log entry — nothing reaches the UI unchecked.
8. Investigator selects a finding to highlight its exact path in the graph, exports a court-ready PDF, and can ask follow-up questions in the same session.

---

## 5. Architecture

Kept intentionally leaner than a full multi-agent stack — enough separation for auditability and swappability, not so much that a 6-person team can't finish it in time.

| Layer | Responsibility |
|---|---|
| API & Orchestration | Session/case lifecycle, routes requests between services |
| Ingestion & Entity Resolution | Parses structured (CDR/IPDR/financial CSVs) and unstructured (FIR text, OCR'd scans) input; multilingual NER; alias/dedup resolution |
| Traversal Service | Iterative, depth-limited graph exploration with cycle protection |
| Graph Service | Knowledge-graph node/relationship construction with provenance metadata on every edge |
| Link-Prediction Service | GNN model (GraphSAGE) surfacing probable undocumented connections, confidence-scored |
| Pattern Analysis Service | Single, final AI pass producing structured, evidence-linked findings |
| Explainability Layer | Attaches source doc + sentence span + confidence to every entity/edge/prediction |
| Compliance & Guardrail Layer | RBAC check, PII redaction, "predicted vs. confirmed" enforcement, immutable audit log — the single most important trust feature in the whole system |
| Federation Gateway | Exchanges hashed entity matches/graph fragments across agencies or countries, never raw data |
| Frontend | Live graph visualization, findings view, evidence drill-down, offline-capable mobile companion |

**Data layer:** PostgreSQL (structured records, cases, audit logs) + Neo4j (live investigation graph) + a vector store (RAG over source documents for grounded, cited answers) + object storage for scanned evidence.

**AI layer:** LLM used for query understanding and final pattern interpretation, with a safe graph-only fallback if unavailable. Architecture is LLM-agnostic by design (cloud API for hackathon speed, swappable to an on-prem model for production — directly answers the "can this run air-gapped" question).

---

## 6. Measurable Success Metrics

Carried over from AarohAI because a judge can actually score against numbers, not adjectives.

| Goal | Metric |
|---|---|
| Reduce time to identify a network | Days (manual) → minutes (demo target: <5 min on a 500-node synthetic dataset) |
| Surface hidden/non-obvious links | ≥80% recall on "planted" hidden relationships in synthetic test data |
| Identify key players correctly | ≥85% precision@5 on centrality-ranked suspects vs. ground truth |
| Explainability | 100% of AI-surfaced links carry a provenance trail (source + confidence) |
| Legal/compliance readiness | Full, tamper-evident audit log of every query, access, and inference |

---

## 7. Feasibility & 36–45 Hour Build Scope

Honesty about what's really buildable is itself a differentiator — judges notice overclaiming.

**Build in the hackathon (must-have, ~65% of time):**
- Synthetic dataset generator (300–500 entities, 5–6 planted hidden networks) — sidesteps any real-data ethics objection
- Case workspace, NL query intake, entity resolution against the synthetic store
- Depth-limited traversal into Neo4j with live graph visualization
- Single AI pass producing confidence-scored findings + mandatory counter-evidence
- Hash-based provenance field on every edge (cheap to build, high judge-impact)
- Compliance Guardrail v1: RBAC stub + "predicted vs. confirmed" label enforcement + audit log
- A heuristic stand-in for link prediction (shared second-degree neighbors), clearly labelled as a v1 approximation of the full GNN

**Should build (~20% of time):**
- Basic Hindi NER on a sample FIR (even partial — enough to show the multilingual story is real, not just a slide)
- One-click PDF report generation
- Simple GraphSAGE trained on the synthetic graph for a real (not just heuristic) predicted-link demo moment

**Nice-to-have if time remains (~15%):**
- Lightweight mobile screen showing offline lookup of a cached subgraph
- Federated query demo between two mock "agency" databases exchanging only hashes

**Explicitly roadmap, not built:** full CCTNS/ICJS integration, production on-prem LLM fine-tuning, cross-state federated learning at scale, full multilingual coverage beyond one demo language.

---

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React (or Next.js), Tailwind CSS, a graph visualization library (Cytoscape.js/Sigma.js) |
| Backend | Python, FastAPI, WebSockets |
| Structured data | PostgreSQL |
| Graph data | Neo4j (+ Graph Data Science library for centrality/community detection) |
| Vector store | Qdrant or pgvector (RAG grounding for cited NL answers) |
| Link prediction | PyTorch Geometric — GraphSAGE (v1 heuristic → full GNN post-hackathon) |
| NLP/NER | spaCy + IndicNER/IndicBERT for Hindi/regional-language extraction |
| AI/LLM | Cloud LLM for hackathon speed; architecture swappable to on-prem Llama/Mistral for production |
| Security | Parameterized queries, hash-anchored provenance, RBAC stub, environment-scoped credentials |
| Mobile (roadmap-adjacent) | React Native/Flutter, offline-first sync |

---

## 9. Ethical, Legal & Security Safeguards

- Prototype uses **synthetic data exclusively** — no real case data at any point.
- Every finding is explicitly framed as an **investigative lead, never a determination of guilt**; predicted links are structurally blocked from being labelled "confirmed."
- Every AI finding carries a confidence score and mandatory counter-evidence to guard against confirmation bias and wrongful profiling.
- Grounded in **DPDP Act 2023** (data minimisation, purpose limitation), **IT Act 2000 §65B** (evidentiary standard for electronic records), and chain-of-custody requirements for anything exported to a chargesheet.
- Federated exchange shares only **hashed matches or graph fragments — never raw PII** — across agencies or borders.
- All data access is parameterized, logged, and traceable; credentials/API keys remain server-side.

---

## 10. Impact

**National:** Cuts multi-hop investigation time from days to minutes; improves chargesheet quality with traceable findings; extends capability to under-resourced rural units via the offline mobile companion; supports inter-state coordination through the federated layer without breaching jurisdictional data boundaries.

**Global:** The core engine is data-source agnostic — deployable by any country's police/justice ministry by swapping connectors, not rebuilding. The federated architecture mirrors recognized data-exchange models (INTERPOL's secure network, US NIEM standard), and directly supports investigation of transnational crime (trafficking, cyber fraud, narcotics) that no single-country tool can address alone.

---

## 11. Conclusion

TRINETRA-X reframes SIH26189 from a "graph + chatbot" demo — the pattern judges have already seen — into a trustworthy, Indian-context-aware, court-defensible investigation platform. It keeps a scope that six people can actually finish in 36–45 hours while carrying the two things that separate a 6/10 SIH submission from a 10/10 one: a structural guardrail against AI hallucination (not just a UI label), and honest grounding in the specific legal, linguistic, and operational realities of Indian law enforcement.

*Prepared for Smart India Hackathon 2026 — Problem Statement SIH26189*
