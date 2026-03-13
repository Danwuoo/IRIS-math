# Goals and Success Criteria

**Document Type:** Active Companion Authority  
**Scope:** Program goals, success surfaces, time-horizon success criteria, and red-line failure conditions for IRIS-math v2  
**Boundary:** This document defines what counts as success. It does not replace metric names, gate semantics, or profile metadata defined elsewhere.

---

## 0. Purpose and Authority

IRIS-math v2 is not defined by benchmark score alone.

This document exists to make the success model explicit:

- what core capabilities the system must possess,
- what system properties must remain stable while those capabilities grow,
- what external outcomes are meaningful,
- which apparent wins still count as failure.

Benchmark families remain outcome surfaces, not architecture sources.

---

## 1. Three-Layer Success Model

### 1.1 Layer 1: Core Capability Goals

IRIS-math v2 must improve along these core capability axes:

1. **Document-native reading**  
   Read mathematical content from PDF, DOCX, images, scanned notes, tables, and diagrams after canonical normalization.

2. **Precise mathematical representation**  
   Build faithful problem frames, symbol tables, constraint structure, and source-grounded anchors.

3. **Controllable strategy induction**  
   Produce multiple plausible proof / program frontiers instead of collapsing into one benchmark-shaped script.

4. **Verifier-grounded reasoning**  
   Let local validity, proof-gap detection, contradiction probes, and formal bridges influence outcome and recovery.

5. **Targeted recovery after failure**  
   Route failure credit to the right levels and recover without blind rerun behavior.

6. **Cross-modal theorem / problem understanding**  
   Use text, formulas, diagrams, and document context as one reasoning substrate.

### 1.2 Layer 2: System Goals

The program must also preserve system-level properties:

1. **Scalable institutions** across `P1` through `P4`
2. **Benchmark-aware but not benchmark-locked** training and evaluation
3. **Stable contamination and provenance governance**
4. **Long-context proof and document handling**
5. **Retrieval and lemma reuse with applicability audit**
6. **Failure-first observability**, not hidden end-to-end luck

### 1.3 Layer 3: External Outcome Goals

Meaningful visible outcomes include:

- strong performance on outcome-facing benchmark families such as `AIMO`, `Omni-MATH`, `miniF2F`, and `FrontierMath`,
- strong document-grounded theorem tracing and proof repair behavior,
- visible gains in cross-modal mathematical reasoning,
- frontier-style performance that remains compatible with verifier evidence and contamination discipline.

These outcomes matter only when supported by Layer 1 and Layer 2 evidence.

---

## 2. Time-Horizon Success Criteria

### 2.1 Near-Term Success

Near-term success means the program can reliably establish institutions:

- `P1` can run the full data / provenance / verifier / contamination / regression loop,
- benchmark tiering is explicit and auditable,
- document parsing produces useful State IR-aligned signals,
- verifier evidence is usable enough to shape failure routing,
- strict held-out leakage is monitored.

This is institution success, not final-model success.

### 2.2 Mid-Term Success

Mid-term success means the system can exploit those institutions at larger scale:

- `P2` or `P3` show stronger strategy diversity and theorem reuse,
- document robustness survives OCR / layout / reformulation stress,
- formal or semi-formal signals produce measurable verifier gains,
- failure-credit distributions remain meaningful under harder tasks,
- benchmark gains are accompanied by homologous held-out gains and stable contamination controls.

### 2.3 Long-Term Success

Long-term success means frontier scaling is justified:

- `P3` and `P4` exhibit non-trivial long-context proof handling,
- verifier evidence is strong enough to support frontier claims,
- strict held-out benchmark families remain genuinely informative,
- the system shows broad mathematical competence across document-native, proof-bearing, and cross-modal tasks,
- external outcomes remain aligned with internal diagnostics rather than replacing them.

---

## 3. Success Claims Must Name Their Surface

Any major program claim should specify which success surface it belongs to:

- **Capability claim**
- **System claim**
- **Outcome claim**

Claims are incomplete if they report outcome without saying whether capability and system evidence moved in the same direction.

---

## 4. Benchmark Outcome Rule

Benchmark families may:

- show external progress,
- stress capability limits,
- reveal benchmark-family-specific weaknesses.

Benchmark families may not:

- define the architecture,
- waive contamination discipline,
- substitute for verifier-grounded evidence,
- replace document-native reasoning goals.

The architecture source remains the active contracts, not the leaderboard.

---

## 5. Red Lines: High Score but Failure

The following cases still count as failure even if headline score improves:

1. **Contamination-blind gains**  
   Score rises while train-visible overlap or strict held-out leakage remains unresolved.

2. **Verifier collapse**  
   Benchmark accuracy rises while false accept behavior, calibration, or proof validity materially degrades.

3. **Failure-credit collapse**  
   Outcome rises but `failure.credit` becomes uninformative or degenerate.

4. **Document-grounding collapse**  
   The system scores well on short clean tasks but loses document-native grounding, OCR robustness, or diagram handling.

5. **Benchmark lock-in**  
   Gains are visible on one benchmark family but do not transfer to homologous held-out or reformulated variants.

6. **Archive compatibility dominance**  
   Legacy ARC compatibility probes improve while active math-native objectives stagnate or regress.

---

## 6. Relationship to Other Documents

- `docs/07_Data_Constitution.md` defines what data may be used and under what contamination controls.
- `docs/15_Benchmark_Registry_and_Tiering_Playbook.md` defines benchmark-family-specific tiering and forbidden uses.
- `docs/16_Verifier_and_Formalization_Stack.md` defines verifier evidence and false accept / false reject handling.
- `docs/18_Optimization_and_Learning_Contract.md` defines how learning pressure may be applied while preserving level-addressable capability growth.
- `docs/19_Runtime_and_Task_Adjudication_Semantics.md` defines how accepted outcomes are adjudicated across answer-bearing, proof-bearing, and formalization-bearing task families.
- `docs/17_Scaling_Promotion_and_Readiness.md` defines when larger profiles are justified.
