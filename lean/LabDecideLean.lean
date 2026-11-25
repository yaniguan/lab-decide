/-!
# LabDecideLean: Formal L6 Conformal Calibration & Stepped Frontier Trust Regime

Self-contained Lean 4 (`v4.36.0`) formalization for `lab-decide` (`L6`), synchronized
with the central 15-repository Auto-Lab Trust Regime (`AutoLabLean`):

1. **Finite-Sample Split Conformal Calibration & kNN Trust Boundary (`ConformalQuery`, `checkDecide`)**:
   - Exact finite-sample order-statistic rank:
     `conformalRank n alphaPermille = ceil((n + 1) * (1000 - alphaPermille) / 1000)`.
   - Proves exact ceiling Sandwich bounds (`conformalRank_exact_lower`, `conformalRank_exact_upper`)
     ruling out floating-point rank drift.
   - Never silently clamps `k > n`: undersized calibration sets MUST abstain (`undersized_calib_always_abstains`).
   - Never extrapolates outside the calibrated kNN support (`out_of_domain_always_abstains`).
   - Enforces ordered intervals within physical solubility bounds `[physMinMilli, physMaxMilli]`.
2. **Stepped Annular Frontier Calibration (`FrontierProbeSpec`, `checkFrontierAction`)**:
   - Enforces both lower and upper annular shell bounds `tau < d(x) <= frontierOuterRadius tau eta`,
     micro-volume cap (`<= 10.0 uL`), direct-stock requirement (`isDirectBuy = true`), and
     calibration-only status (`usedAsPointPrediction = false`).
3. **Certified-Support Chaining (`SupportChainHop`, `checkSupportHop`, `checkCertifiedChain`)**:
   - Proves that multi-hop `Chem-RRT` branches cannot chain unassayed or quarantined probes
     (`unassayed_probe_blocks_chaining`, `unassayed_hop_blocks_chain`), and bounds total branch
     reach by `k * frontierOuterRadius tau eta` (`certified_chain_annular_span`).
-/
namespace LabDecideLean

/-- Structured boundary violation reporting the repository, rule identifier, and detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

def RegimeViolation.format (v : RegimeViolation) : String :=
  s!"[{v.repo} :: {v.rule}] {v.detail}"

/-- Return the first violation in a list of `(holds, violation)` checks. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, v⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-! ## 1. Finite-Sample Conformal Calibration & Applicability-Domain Gate -/

/-- Exact integer ceiling `ceil((n + 1) * (1000 - alphaPermille) / 1000)`. -/
def conformalRank (n alphaPermille : Nat) : Nat :=
  ((n + 1) * (1000 - alphaPermille) + 999) / 1000

/-- Fixed-point summary of a conformal interval and applicability-domain query. -/
structure ConformalQuery where
  nCalib           : Nat
  alphaPermille    : Nat
  knnDistPermille  : Nat
  tauDistPermille  : Nat
  loMilli          : Int
  hiMilli          : Int
  physMinMilli     : Int
  physMaxMilli     : Int
  deriving DecidableEq, Repr

/-- Declarative specification of a valid, in-domain conformal decision. -/
structure ConformalDecisionOK (q : ConformalQuery) : Prop where
  alphaValid      : 0 < q.alphaPermille ∧ q.alphaPermille < 1000
  calibNonEmpty   : 0 < q.nCalib
  rankWithinCalib : conformalRank q.nCalib q.alphaPermille ≤ q.nCalib
  insideTrustZone : q.knnDistPermille ≤ q.tauDistPermille
  intervalOrdered : q.loMilli ≤ q.hiMilli
  withinPhysMin   : q.physMinMilli ≤ q.loMilli
  withinPhysMax   : q.hiMilli ≤ q.physMaxMilli

/-- Decidable guardrail checker for `lab-decide` conformal predictions. -/
def checkDecide (q : ConformalQuery) : Option RegimeViolation :=
  firstViolation
    [ (decide (0 < q.alphaPermille ∧ q.alphaPermille < 1000),
        { repo := "lab-decide", rule := "alpha_in_open_unit_interval",
          detail := "alphaPermille must lie strictly in (0, 1000)" }),
      (decide (0 < q.nCalib),
        { repo := "lab-decide", rule := "nonempty_calibration_set",
          detail := "calibration set size nCalib must be positive" }),
      (decide (conformalRank q.nCalib q.alphaPermille ≤ q.nCalib),
        { repo := "lab-decide", rule := "finite_sample_conformal_rank",
          detail := "ceil((n+1)*(1-alpha)) exceeds nCalib; interval is infinite -> must ABSTAIN" }),
      (decide (q.knnDistPermille ≤ q.tauDistPermille),
        { repo := "lab-decide", rule := "knn_trust_boundary",
          detail := "candidate distance exceeds calibrated kNN trust threshold -> must ABSTAIN" }),
      (decide (q.loMilli ≤ q.hiMilli),
        { repo := "lab-decide", rule := "interval_well_ordered",
          detail := "lower conformal bound exceeds upper bound" }),
      (decide (q.physMinMilli ≤ q.loMilli ∧ q.hiMilli ≤ q.physMaxMilli),
        { repo := "lab-decide", rule := "physical_solubility_bounds",
          detail := "prediction interval violates physical solubility range [physMin, physMax]" }) ]

/-- Soundness and completeness of `checkDecide` wrt `ConformalDecisionOK`. -/
theorem checkDecide_ok_iff (q : ConformalQuery) :
    checkDecide q = none ↔ ConformalDecisionOK q := by
  unfold checkDecide
  simp only [firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq]
  constructor
  · rintro ⟨h1, h2, h3, h4, h5, h6a, h6b⟩
    exact ⟨h1, h2, h3, h4, h5, h6a, h6b⟩
  · rintro ⟨h1, h2, h3, h4, h5, h6a, h6b⟩
    exact ⟨h1, h2, h3, h4, h5, h6a, h6b⟩

/-- Hard boundary theorem 1: Out-of-domain molecules (`knnDist > tauDist`) are never accepted. -/
theorem out_of_domain_always_abstains (q : ConformalQuery)
    (hood : q.tauDistPermille < q.knnDistPermille) :
    checkDecide q ≠ none := by
  intro hok
  have hd := ((checkDecide_ok_iff q).mp hok).insideTrustZone
  omega

/-- Hard boundary theorem 2: Undersized calibration sets (`k > n`) are never silently clamped. -/
theorem undersized_calib_always_abstains (q : ConformalQuery)
    (hsmall : q.nCalib < conformalRank q.nCalib q.alphaPermille) :
    checkDecide q ≠ none := by
  intro hok
  have hr := ((checkDecide_ok_iff q).mp hok).rankWithinCalib
  omega

/-- Whenever `alphaPermille < 1000`, the finite-sample conformal rank is strictly `>= 1`. -/
theorem conformalRank_pos (n alphaPermille : Nat) (ha : alphaPermille < 1000) :
    1 ≤ conformalRank n alphaPermille := by
  unfold conformalRank
  have hpos : 0 < (n + 1) * (1000 - alphaPermille) :=
    Nat.mul_pos (Nat.succ_pos n) (by omega)
  omega

/-- Exact ceiling lower bound: `conformalRank` never under-covers below `(n + 1) * (1 - alpha)`. -/
theorem conformalRank_exact_lower (n alphaPermille : Nat) :
    (n + 1) * (1000 - alphaPermille) ≤ 1000 * conformalRank n alphaPermille := by
  unfold conformalRank
  omega

/-- Exact ceiling minimality: `conformalRank - 1` is strictly below `(n + 1) * (1 - alpha)`. -/
theorem conformalRank_exact_upper (n alphaPermille : Nat) (ha : alphaPermille < 1000) :
    1000 * (conformalRank n alphaPermille - 1) < (n + 1) * (1000 - alphaPermille) := by
  have hpos : 1 ≤ conformalRank n alphaPermille := conformalRank_pos n alphaPermille ha
  unfold conformalRank at hpos ⊢
  omega

/-- Accepted conformal intervals always lie ordered within `[physMinMilli, physMaxMilli]`. -/
theorem physical_bounds_respected (q : ConformalQuery)
    (hok : checkDecide q = none) :
    q.physMinMilli ≤ q.loMilli ∧ q.loMilli ≤ q.hiMilli ∧ q.hiMilli ≤ q.physMaxMilli := by
  have h := (checkDecide_ok_iff q).mp hok
  exact ⟨h.withinPhysMin, h.intervalOrdered, h.withinPhysMax⟩

/-! ## 2. Stepped Annular Frontier Calibration (`Chem-RRT` Stepping-Stone Guard) -/

inductive CampaignMode where
  | exploitation
  | frontierCalibration
  deriving DecidableEq, Repr

/-- Specification of an exploitation or annular stepping-stone frontier action. -/
structure FrontierProbeSpec where
  mode                  : CampaignMode
  knnDistPermille       : Nat
  tauPermille           : Nat
  etaStepPermille       : Nat
  maxEtaPermille        : Nat
  assayVolTenthUL       : Nat
  microVolCapTenthUL    : Nat
  isDirectBuy           : Bool
  usedAsPointPrediction : Bool
  deriving DecidableEq, Repr

/-- Outer radius `tau + floor(tau * etaPermille / 1000)` of the stepping-stone shell. -/
def frontierOuterRadius (tau etaPermille : Nat) : Nat :=
  tau + (tau * etaPermille) / 1000

theorem frontierOuterRadius_ge_tau (tau etaPermille : Nat) :
    tau ≤ frontierOuterRadius tau etaPermille := by
  unfold frontierOuterRadius
  omega

def checkFrontierAction (p : FrontierProbeSpec) : Option RegimeViolation :=
  match p.mode with
  | .exploitation =>
    firstViolation
      [ (decide (p.knnDistPermille ≤ p.tauPermille),
          { repo := "lab-decide", rule := "exploitation_inside_tau",
            detail := "exploitation run exceeds current calibrated trust radius tau" }) ]
  | .frontierCalibration =>
    firstViolation
      [ (decide (p.etaStepPermille ≤ p.maxEtaPermille),
          { repo := "lab-decide", rule := "bounded_annular_shell_width",
            detail := "frontier stepping-stone expansion ratio eta exceeds safe shell cap" }),
        (decide (p.tauPermille < p.knnDistPermille),
          { repo := "lab-decide", rule := "strictly_outside_tau_shell",
            detail := "frontier calibration probe must lie strictly outside current calibrated radius tau" }),
        (decide (p.knnDistPermille ≤ frontierOuterRadius p.tauPermille p.etaStepPermille),
          { repo := "lab-decide", rule := "within_stepping_stone_shell",
            detail := "candidate lies in deep OOD beyond the (1 + eta) * tau stepping-stone shell" }),
        (decide (p.assayVolTenthUL ≤ p.microVolCapTenthUL),
          { repo := "lab-rehearsal", rule := "frontier_micro_volume_cap",
            detail := "frontier calibration probe exceeds micro-scale reagent volume cap" }),
        (p.isDirectBuy,
          { repo := "buy-or-make", rule := "frontier_requires_direct_stock",
            detail := "multi-step synthesis forbidden outside calibrated trust zone; probe stock leaves first" }),
        (!p.usedAsPointPrediction,
          { repo := "lab-decide", rule := "frontier_calibration_only",
            detail := "frontier probe cannot be consumed as a calibrated point prediction prior to assay" }) ]

/-- Interior candidates (`knnDist <= tau`) cannot be misclassified as frontier stepping stones. -/
theorem interior_rejected_as_frontier_probe (p : FrontierProbeSpec)
    (hm : p.mode = .frontierCalibration)
    (hin : p.knnDistPermille ≤ p.tauPermille) :
    checkFrontierAction p ≠ none := by
  intro hok
  simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
             and_true, decide_eq_true_eq] at hok
  rcases hok with ⟨_, hlow, _, _, _, _⟩
  omega

/-- Accepted frontier calibration probes lie strictly inside `(tau, frontierOuterRadius tau eta]`. -/
theorem frontier_calibration_strictly_annular (p : FrontierProbeSpec)
    (hm : p.mode = .frontierCalibration)
    (hok : checkFrontierAction p = none) :
    p.tauPermille < p.knnDistPermille ∧
    p.knnDistPermille ≤ frontierOuterRadius p.tauPermille p.etaStepPermille := by
  simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
             and_true, decide_eq_true_eq] at hok
  rcases hok with ⟨_, hlow, hhigh, _, _, _⟩
  exact ⟨hlow, hhigh⟩

/-- Deep OOD candidates beyond `(1 + eta) * tau` are blocked in ALL campaign modes. -/
theorem deep_ood_always_blocked (p : FrontierProbeSpec)
    (hdeep : frontierOuterRadius p.tauPermille p.etaStepPermille < p.knnDistPermille) :
    checkFrontierAction p ≠ none := by
  intro hok
  cases hm : p.mode with
  | exploitation =>
    simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
               and_true, decide_eq_true_eq] at hok
    unfold frontierOuterRadius at hdeep
    omega
  | frontierCalibration =>
    simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
               and_true, decide_eq_true_eq] at hok
    rcases hok with ⟨_, _, hshell, _, _, _⟩
    omega

/-- Any candidate accepted outside `tau` is provably a micro-scale, direct-buy calibration probe. -/
theorem accepted_outside_tau_is_bounded_probe (p : FrontierProbeSpec)
    (hout : p.tauPermille < p.knnDistPermille)
    (hok : checkFrontierAction p = none) :
    p.mode = .frontierCalibration ∧
    p.knnDistPermille ≤ frontierOuterRadius p.tauPermille p.etaStepPermille ∧
    p.assayVolTenthUL ≤ p.microVolCapTenthUL ∧
    p.isDirectBuy = true ∧
    p.usedAsPointPrediction = false := by
  cases hm : p.mode with
  | exploitation =>
    simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
               and_true, decide_eq_true_eq] at hok
    omega
  | frontierCalibration =>
    simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
               and_true, decide_eq_true_eq, Bool.not_eq_true'] at hok
    rcases hok with ⟨_, _, hshell, hvol, hbuy, hcal⟩
    exact ⟨rfl, hshell, hvol, hbuy, hcal⟩

/-! ## 3. Multi-Hop `Chem-RRT` Certified-Support Chaining -/

/-- Directed stepping-stone hop requiring wet-lab certification on both parent and probe. -/
structure SupportChainHop where
  probe           : FrontierProbeSpec
  parentCertified : Bool
  probeCertified  : Bool
  deriving DecidableEq, Repr

/-- Validate one stepping-stone hop before promoting `probe` into calibrated support `S_{t+1}`. -/
def checkSupportHop (h : SupportChainHop) : Option RegimeViolation :=
  match checkFrontierAction h.probe with
  | some v => some v
  | none =>
    firstViolation
      [ (h.parentCertified,
          { repo := "lab-attrib", rule := "parent_support_certified",
            detail := "stepping-stone parent node lacks certified wet-lab support" }),
        (decide (h.probe.mode = .frontierCalibration),
          { repo := "lab-decide", rule := "chain_hop_is_frontier_probe",
            detail := "support-expansion hop must execute in frontierCalibration mode" }),
        (h.probeCertified,
          { repo := "lab-attrib", rule := "probe_wetlab_certified_before_chaining",
            detail := "unassayed or quarantined probe cannot be promoted into calibrated support" }) ]

/-- Validate an entire multi-hop `Chem-RRT` stepping-stone branch. -/
def checkCertifiedChain : List SupportChainHop → Option RegimeViolation
  | [] => none
  | hd :: tl =>
    match checkSupportHop hd with
    | some v => some v
    | none => checkCertifiedChain tl

/-- Total cumulative stepping-stone distance (permille) traversed along a branch. -/
def chainTotalSpan : List SupportChainHop → Nat
  | [] => 0
  | hd :: tl => hd.probe.knnDistPermille + chainTotalSpan tl

@[simp] theorem checkSupportHop_ok_iff (h : SupportChainHop) :
    checkSupportHop h = none ↔
      checkFrontierAction h.probe = none ∧
      h.parentCertified = true ∧
      h.probe.mode = .frontierCalibration ∧
      h.probeCertified = true := by
  unfold checkSupportHop
  cases hfa : checkFrontierAction h.probe with
  | some v => simp
  | none =>
    simp [firstViolation_cons, firstViolation_nil, decide_eq_true_eq]

@[simp] theorem checkCertifiedChain_nil : checkCertifiedChain [] = none := rfl

@[simp] theorem checkCertifiedChain_cons (hd : SupportChainHop) (tl : List SupportChainHop) :
    checkCertifiedChain (hd :: tl) = none ↔
      checkSupportHop hd = none ∧ checkCertifiedChain tl = none := by
  cases hh : checkSupportHop hd <;> simp [checkCertifiedChain, hh]

/-- Unassayed or quarantined stepping-stone probes (`probeCertified = false`) can never be chained. -/
theorem unassayed_probe_blocks_chaining (h : SupportChainHop)
    (hun : h.probeCertified = false) :
    checkSupportHop h ≠ none := by
  intro hok
  have hc := ((checkSupportHop_ok_iff h).mp hok).2.2.2
  rw [hun] at hc
  contradiction

/-- Any multi-hop chain containing an unassayed probe is rejected. -/
theorem unassayed_hop_blocks_chain (chain : List SupportChainHop)
    (h : SupportChainHop) (hmem : h ∈ chain) (hun : h.probeCertified = false) :
    checkCertifiedChain chain ≠ none := by
  induction chain with
  | nil => cases hmem
  | cons hd tl ih =>
    intro hok
    rcases (checkCertifiedChain_cons hd tl).mp hok with ⟨hhd, htl⟩
    rcases List.mem_cons.mp hmem with heq | htl_mem
    · rw [heq] at hun
      exact unassayed_probe_blocks_chaining hd hun hhd
    · exact ih htl_mem htl

/-- Every accepted support-chain hop satisfies strict annular bounds, micro-volume cap, and certification. -/
theorem certified_hop_annular_bounds (h : SupportChainHop)
    (hok : checkSupportHop h = none) :
    h.parentCertified = true ∧
    h.probeCertified = true ∧
    h.probe.tauPermille < h.probe.knnDistPermille ∧
    h.probe.knnDistPermille ≤ frontierOuterRadius h.probe.tauPermille h.probe.etaStepPermille ∧
    h.probe.assayVolTenthUL ≤ h.probe.microVolCapTenthUL ∧
    h.probe.isDirectBuy = true ∧
    h.probe.usedAsPointPrediction = false := by
  rcases (checkSupportHop_ok_iff h).mp hok with ⟨hfa, hp, hm, hc⟩
  rcases frontier_calibration_strictly_annular h.probe hm hfa with ⟨hlow, hhigh⟩
  rcases accepted_outside_tau_is_bounded_probe h.probe hlow hfa with ⟨_, _, hvol, hbuy, hcal⟩
  exact ⟨hp, hc, hlow, hhigh, hvol, hbuy, hcal⟩

/-- Telescoping annular span theorem: a certified `k`-hop branch with parameters `(tau, eta)`
    advances strictly more than `k * tau` and at most `k * frontierOuterRadius tau eta`. -/
theorem certified_chain_annular_span (chain : List SupportChainHop) (tau eta : Nat)
    (hok : checkCertifiedChain chain = none)
    (hcfg : ∀ h ∈ chain, h.probe.tauPermille = tau ∧ h.probe.etaStepPermille = eta) :
    chain.length * tau + chain.length ≤ chainTotalSpan chain ∧
    chainTotalSpan chain ≤ chain.length * frontierOuterRadius tau eta := by
  induction chain with
  | nil => simp [chainTotalSpan]
  | cons hd tl ih =>
    rcases (checkCertifiedChain_cons hd tl).mp hok with ⟨hhd, htl⟩
    have hhd_cfg := hcfg hd List.mem_cons_self
    have htl_cfg : ∀ h ∈ tl, h.probe.tauPermille = tau ∧ h.probe.etaStepPermille = eta :=
      fun h hm => hcfg h (List.mem_cons_of_mem hd hm)
    rcases ih htl htl_cfg with ⟨ih_lo, ih_hi⟩
    rcases certified_hop_annular_bounds hd hhd with ⟨_, _, hlow, hhigh, _, _, _⟩
    rw [hhd_cfg.1] at hlow
    rw [hhd_cfg.1, hhd_cfg.2] at hhigh
    simp only [List.length_cons, chainTotalSpan]
    have hmul_lo : (tl.length + 1) * tau = tl.length * tau + tau := Nat.succ_mul tl.length tau
    have hmul_hi : (tl.length + 1) * frontierOuterRadius tau eta =
        tl.length * frontierOuterRadius tau eta + frontierOuterRadius tau eta :=
      Nat.succ_mul tl.length (frontierOuterRadius tau eta)
    omega

/-- Gate a production exploitation run behind a certified `Chem-RRT` branch. -/
def checkExploitAfterChain (chain : List SupportChainHop) (exploit : FrontierProbeSpec)
    (anchorCertified : Bool) : Option RegimeViolation :=
  match checkCertifiedChain chain with
  | some v => some v
  | none =>
    if !anchorCertified then
      some { repo := "lab-attrib", rule := "parent_support_certified",
             detail := "exploitation support anchor is not wet-lab certified" }
    else if exploit.mode != .exploitation then
      some { repo := "lab-decide", rule := "terminal_action_is_exploitation",
             detail := "terminal target run must execute in exploitation mode" }
    else
      checkFrontierAction exploit

theorem exploitation_after_chain_sound (chain : List SupportChainHop)
    (exploit : FrontierProbeSpec) (anchorCertified : Bool)
    (hok : checkExploitAfterChain chain exploit anchorCertified = none) :
    checkCertifiedChain chain = none ∧
    anchorCertified = true ∧
    exploit.mode = .exploitation ∧
    exploit.knnDistPermille ≤ exploit.tauPermille := by
  unfold checkExploitAfterChain at hok
  cases hc : checkCertifiedChain chain with
  | some v => simp [hc] at hok
  | none =>
    simp only [hc] at hok
    cases ha : anchorCertified with
    | false => simp [ha] at hok
    | true =>
      simp only [ha, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hok
      cases hm : exploit.mode with
      | frontierCalibration => simp [hm] at hok
      | exploitation =>
        simp only [hm, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte,
                   checkFrontierAction, firstViolation_cons, firstViolation_nil,
                   and_true, decide_eq_true_eq] at hok
        exact ⟨rfl, rfl, rfl, hok⟩

/-! ## 4. Compile-Time `#guard` Witnesses -/

#guard conformalRank 149 180 == 123

#guard (checkDecide
         { nCalib := 180, alphaPermille := 100, knnDistPermille := 410,
           tauDistPermille := 758, loMilli := -2410, hiMilli := 630,
           physMinMilli := -15000, physMaxMilli := 1750 }).isNone == true

-- Undersized calibration set (`n = 5` at `alpha = 0.10` -> `rank = 6 > 5`) -> ABSTAIN
#guard (checkDecide
         { nCalib := 5, alphaPermille := 100, knnDistPermille := 300,
           tauDistPermille := 758, loMilli := -2000, hiMilli := 500,
           physMinMilli := -15000, physMaxMilli := 1750 }).isSome == true

-- Out-of-domain candidate (`knnDist = 1380 > 758`) -> ABSTAIN
#guard (checkDecide
         { nCalib := 180, alphaPermille := 100, knnDistPermille := 1380,
           tauDistPermille := 758, loMilli := -2410, hiMilli := 630,
           physMinMilli := -15000, physMaxMilli := 1750 }).isSome == true

-- Valid annular stepping-stone probe (`758 < 900 <= 947`, `8.0 uL <= 10.0 uL`, direct buy) -> OK
#guard (checkFrontierAction
         { mode := .frontierCalibration, knnDistPermille := 900, tauPermille := 758,
           etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 80,
           microVolCapTenthUL := 100, isDirectBuy := true,
           usedAsPointPrediction := false }).isNone == true

-- Interior point (`knnDist = 500 <= 758`) rejected as frontier stepping-stone probe -> BLOCKED
#guard (checkFrontierAction
         { mode := .frontierCalibration, knnDistPermille := 500, tauPermille := 758,
           etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 80,
           microVolCapTenthUL := 100, isDirectBuy := true,
           usedAsPointPrediction := false }).isSome == true

-- Deep OOD leap (`1400 > 947`) -> BLOCKED
#guard (checkFrontierAction
         { mode := .frontierCalibration, knnDistPermille := 1400, tauPermille := 758,
           etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 80,
           microVolCapTenthUL := 100, isDirectBuy := true,
           usedAsPointPrediction := false }).isSome == true

-- Production-volume (`25.0 uL > 10.0 uL`) outside tau -> BLOCKED
#guard (checkFrontierAction
         { mode := .frontierCalibration, knnDistPermille := 900, tauPermille := 758,
           etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 250,
           microVolCapTenthUL := 100, isDirectBuy := true,
           usedAsPointPrediction := false }).isSome == true

-- Unassayed probe (`probeCertified := false`) in multi-hop chain -> BLOCKED
#guard (checkCertifiedChain
         [ { probe := { mode := .frontierCalibration, knnDistPermille := 900, tauPermille := 758,
                        etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 80,
                        microVolCapTenthUL := 100, isDirectBuy := true,
                        usedAsPointPrediction := false },
             parentCertified := true,
             probeCertified := false } ]).isSome == true

-- Certified 2-hop chain followed by inside-tau exploitation -> OK
#guard (checkExploitAfterChain
         [ { probe := { mode := .frontierCalibration, knnDistPermille := 900, tauPermille := 758,
                        etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 80,
                        microVolCapTenthUL := 100, isDirectBuy := true,
                        usedAsPointPrediction := false },
             parentCertified := true,
             probeCertified := true },
           { probe := { mode := .frontierCalibration, knnDistPermille := 920, tauPermille := 758,
                        etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 80,
                        microVolCapTenthUL := 100, isDirectBuy := true,
                        usedAsPointPrediction := false },
             parentCertified := true,
             probeCertified := true } ]
         { mode := .exploitation, knnDistPermille := 680, tauPermille := 758,
           etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 500,
           microVolCapTenthUL := 100, isDirectBuy := false,
           usedAsPointPrediction := true }
         true).isNone == true

end LabDecideLean
